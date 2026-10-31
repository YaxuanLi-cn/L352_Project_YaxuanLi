import sys
import numpy as np
import torch
import torch.nn as nn


class HDRFLIPLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.qc = 0.7
		self.qf = 0.5
		self.pc = 0.4
		self.pt = 0.95
		self.tmax = 0.85
		self.tmin = 0.85
		self.eps = 1e-15
	def forward(self, test, reference, pixels_per_degree=(0.7 * 3840 / 0.7) * np.pi / 180, tone_mapper="aces", start_exposure=None, stop_exposure=None):
		reference = torch.clamp(reference, 0, 65536.0)
		test = torch.clamp(test, 0, 65536.0)
		if start_exposure is None or stop_exposure is None:
			c_start, c_stop = compute_start_stop_exposures(reference, tone_mapper, self.tmax, self.tmin)
			if start_exposure is None:
				start_exposure = c_start
			if stop_exposure is None:
				stop_exposure = c_stop
		num_exposures = torch.max(torch.tensor([2.0], requires_grad=False).cuda(), torch.ceil(stop_exposure - start_exposure))
		most_exposures = int(torch.amax(num_exposures, dim=0).item())
		step_size = (stop_exposure - start_exposure) / torch.max(num_exposures - 1, torch.tensor([1.0], requires_grad=False).cuda())
		dim = reference.size()
		all_errors = torch.zeros(size=(dim[0], most_exposures, dim[2], dim[3])).cuda()
		for i in range(0, most_exposures):
			exposure = start_exposure + i * step_size
			reference_tone_mapped = tone_map(reference, tone_mapper, exposure)
			test_tone_mapped = tone_map(test, tone_mapper, exposure)
			reference_opponent = color_space_transform(reference_tone_mapped, 'linrgb2ycxcz')
			test_opponent = color_space_transform(test_tone_mapped, 'linrgb2ycxcz')
			all_errors[:, i, :, :] = compute_ldrflip(
				test_opponent, reference_opponent, pixels_per_degree,
				self.qc, self.qf, self.pc, self.pt, self.eps
			).squeeze(1)
		hdrflip_error = torch.amax(all_errors, dim=1, keepdim=True)
		return torch.mean(hdrflip_error)


class LDRFLIPLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.qc = 0.7
		self.qf = 0.5
		self.pc = 0.4
		self.pt = 0.95
		self.eps = 1e-15
	def forward(self, test, reference, pixels_per_degree=(0.7 * 3840 / 0.7) * np.pi / 180):
		reference = torch.clamp(reference, 0, 1)
		test = torch.clamp(test, 0, 1)
		reference_opponent = color_space_transform(reference, 'srgb2ycxcz')
		test_opponent = color_space_transform(test, 'srgb2ycxcz')
		deltaE = compute_ldrflip(test_opponent, reference_opponent, pixels_per_degree, self.qc, self.qf, self.pc, self.pt, self.eps)
		return torch.mean(deltaE)


def compute_ldrflip(test, reference, pixels_per_degree, qc, qf, pc, pt, eps):
	s_a, radius_a = generate_spatial_filter(pixels_per_degree, 'A')
	s_rg, radius_rg = generate_spatial_filter(pixels_per_degree, 'RG')
	s_by, radius_by = generate_spatial_filter(pixels_per_degree, 'BY')
	radius = max(radius_a, radius_rg, radius_by)
	filtered_reference = spatial_filter(reference, s_a, s_rg, s_by, radius)
	filtered_test = spatial_filter(test, s_a, s_rg, s_by, radius)
	preprocessed_reference = hunt_adjustment(color_space_transform(filtered_reference, 'linrgb2lab'))
	preprocessed_test = hunt_adjustment(color_space_transform(filtered_test, 'linrgb2lab'))
	deltaE_hyab = hyab(preprocessed_reference, preprocessed_test, eps)
	power_deltaE_hyab = torch.pow(deltaE_hyab, qc)
	hunt_adjusted_green = hunt_adjustment(color_space_transform(torch.tensor([[[0.0]], [[1.0]], [[0.0]]]).unsqueeze(0), 'linrgb2lab'))
	hunt_adjusted_blue = hunt_adjustment(color_space_transform(torch.tensor([[[0.0]], [[0.0]], [[1.0]]]).unsqueeze(0), 'linrgb2lab'))
	cmax = torch.pow(hyab(hunt_adjusted_green, hunt_adjusted_blue, eps), qc).item()
	deltaE_c = redistribute_errors(power_deltaE_hyab, cmax, pc, pt)
	ref_y = (reference[:, 0:1, :, :] + 16) / 116
	test_y = (test[:, 0:1, :, :] + 16) / 116
	edges_reference = feature_detection(ref_y, pixels_per_degree, 'edge')
	points_reference = feature_detection(ref_y, pixels_per_degree, 'point')
	edges_test = feature_detection(test_y, pixels_per_degree, 'edge')
	points_test = feature_detection(test_y, pixels_per_degree, 'point')
	deltaE_f = torch.max(
		torch.abs(torch.norm(edges_reference, dim=1, keepdim=True) - torch.norm(edges_test, dim=1, keepdim=True)),
		torch.abs(torch.norm(points_test, dim=1, keepdim=True) - torch.norm(points_reference, dim=1, keepdim=True))
	)
	deltaE_f = torch.clamp(deltaE_f, min=eps)                               
	deltaE_f = torch.pow(((1 / np.sqrt(2)) * deltaE_f), qf)
	return torch.pow(deltaE_c, 1 - deltaE_f)


def tone_map(img, tone_mapper, exposure):
	x = (2 ** exposure) * img
	if tone_mapper == "reinhard":
		lum_coeff_r = 0.2126
		lum_coeff_g = 0.7152
		lum_coeff_b = 0.0722
		Y = x[:, 0:1, :, :] * lum_coeff_r + x[:, 1:2, :, :] * lum_coeff_g + x[:, 2:3, :, :] * lum_coeff_b
		return torch.clamp(torch.div(x, 1 + Y), 0.0, 1.0)
	if tone_mapper == "hable":
		A = 0.15
		B = 0.50
		C = 0.10
		D = 0.20
		E = 0.02
		F = 0.30
		k0 = A * F - A * E
		k1 = C * B * F - B * E
		k2 = 0
		k3 = A * F
		k4 = B * F
		k5 = D * F * F
		W = 11.2
		nom = k0 * torch.pow(W, torch.tensor([2.0]).cuda()) + k1 * W + k2
		denom = k3 * torch.pow(W, torch.tensor([2.0]).cuda()) + k4 * W + k5
		white_scale = torch.div(denom, nom)                       
		k0 = 4 * k0 * white_scale
		k1 = 2 * k1 * white_scale
		k2 = k2 * white_scale
		k3 = 4 * k3
		k4 = 2 * k4
	else:
		k0 = 0.6 * 0.6 * 2.51
		k1 = 0.6 * 0.03
		k2 = 0
		k3 = 0.6 * 0.6 * 2.43
		k4 = 0.6 * 0.59
		k5 = 0.14
	x2 = torch.pow(x, 2)
	nom = k0 * x2 + k1 * x + k2
	denom = k3 * x2 + k4 * x + k5
	denom = torch.where(torch.isinf(denom), torch.Tensor([1.0]).cuda(), denom)                                                                                                              
	y = torch.div(nom, denom)
	return torch.clamp(y, 0.0, 1.0)


def compute_start_stop_exposures(reference, tone_mapper, tmax, tmin):
	if tone_mapper == "reinhard":
		k0 = 0
		k1 = 1
		k2 = 0
		k3 = 0
		k4 = 1
		k5 = 1
		x_max = tmax * k5 / (k1 - tmax * k4)
		x_min = tmin * k5 / (k1 - tmin * k4)
	elif tone_mapper == "hable":
		A = 0.15
		B = 0.50
		C = 0.10
		D = 0.20
		E = 0.02
		F = 0.30
		k0 = A * F - A * E
		k1 = C * B * F - B * E
		k2 = 0
		k3 = A * F
		k4 = B * F
		k5 = D * F * F
		W = 11.2
		nom = k0 * torch.pow(W, torch.tensor([2.0]).cuda()) + k1 * W + k2
		denom = k3 * torch.pow(W, torch.tensor([2.0]).cuda()) + k4 * W + k5
		white_scale = torch.div(denom, nom)                       
		k0 = 4 * k0 * white_scale
		k1 = 2 * k1 * white_scale
		k2 = k2 * white_scale
		k3 = 4 * k3
		k4 = 2 * k4
		c0 = (k1 - k4 * tmax) / (k0 - k3 * tmax)
		c1 = (k2 - k5 * tmax) / (k0 - k3 * tmax)
		x_max = - 0.5 * c0 + torch.sqrt(((torch.tensor([0.5]).cuda() * c0) ** 2) - c1)
		c0 = (k1 - k4 * tmin) / (k0 - k3 * tmin)
		c1 = (k2 - k5 * tmin) / (k0 - k3 * tmin)
		x_min = - 0.5 * c0 + torch.sqrt(((torch.tensor([0.5]).cuda() * c0) ** 2) - c1)
	else:
		k0 = 0.6 * 0.6 * 2.51
		k1 = 0.6 * 0.03
		k2 = 0
		k3 = 0.6 * 0.6 * 2.43
		k4 = 0.6 * 0.59
		k5 = 0.14
		c0 = (k1 - k4 * tmax) / (k0 - k3 * tmax)
		c1 = (k2 - k5 * tmax) / (k0 - k3 * tmax)
		x_max = - 0.5 * c0 + torch.sqrt(((torch.tensor([0.5]).cuda() * c0) ** 2) - c1)
		c0 = (k1 - k4 * tmin) / (k0 - k3 * tmin)
		c1 = (k2 - k5 * tmin) / (k0 - k3 * tmin)
		x_min = - 0.5 * c0 + torch.sqrt(((torch.tensor([0.5]).cuda() * c0) ** 2) - c1)
	lum_coeff_r = 0.2126
	lum_coeff_g = 0.7152
	lum_coeff_b = 0.0722
	Y_reference = reference[:, 0:1, :, :] * lum_coeff_r + reference[:, 1:2, :, :] * lum_coeff_g + reference[:, 2:3, :, :] * lum_coeff_b
	Y_hi = torch.amax(Y_reference, dim=(2, 3), keepdim=True)
	start_exposure = torch.log2(x_max / Y_hi)
	dim = Y_reference.size()
	Y_ref = Y_reference.view(dim[0], dim[1], dim[2]*dim[3])
	Y_lo = torch.median(Y_ref, dim=2).values.unsqueeze(2).unsqueeze(3)
	Y_lo = torch.clamp(Y_lo, min=torch.finfo(torch.float32).eps)                                                                        
	stop_exposure = torch.log2(x_min / Y_lo)
	return start_exposure, stop_exposure


def generate_spatial_filter(pixels_per_degree, channel):
	a1_A = 1
	b1_A = 0.0047
	a2_A = 0
	b2_A = 1e-5                       
	a1_rg = 1
	b1_rg = 0.0053
	a2_rg = 0
	b2_rg = 1e-5                       
	a1_by = 34.1
	b1_by = 0.04
	a2_by = 13.5
	b2_by = 0.025
	if channel == "A":                  
		a1 = a1_A
		b1 = b1_A
		a2 = a2_A
		b2 = b2_A
	elif channel == "RG":                 
		a1 = a1_rg
		b1 = b1_rg
		a2 = a2_rg
		b2 = b2_rg
	elif channel == "BY":                   
		a1 = a1_by
		b1 = b1_by
		a2 = a2_by
		b2 = b2_by
	max_scale_parameter = max([b1_A, b2_A, b1_rg, b2_rg, b1_by, b2_by])
	r = np.ceil(3 * np.sqrt(max_scale_parameter / (2 * np.pi**2)) * pixels_per_degree)
	r = int(r)
	deltaX = 1.0 / pixels_per_degree
	x, y = np.meshgrid(range(-r, r + 1), range(-r, r + 1))
	z = (x * deltaX)**2 + (y * deltaX)**2
	g = a1 * np.sqrt(np.pi / b1) * np.exp(-np.pi**2 * z / b1) + a2 * np.sqrt(np.pi / b2) * np.exp(-np.pi**2 * z / b2)
	g = g / np.sum(g)
	g = torch.Tensor(g).unsqueeze(0).unsqueeze(0).cuda()
	return g, r


def spatial_filter(img, s_a, s_rg, s_by, radius):
	dim = img.size()
	img_pad = torch.zeros((dim[0], dim[1], dim[2] + 2 * radius, dim[3] + 2 * radius), device='cuda')
	img_pad[:, 0:1, :, :] = nn.functional.pad(img[:, 0:1, :, :], (radius, radius, radius, radius), mode='replicate')
	img_pad[:, 1:2, :, :] = nn.functional.pad(img[:, 1:2, :, :], (radius, radius, radius, radius), mode='replicate')
	img_pad[:, 2:3, :, :] = nn.functional.pad(img[:, 2:3, :, :], (radius, radius, radius, radius), mode='replicate')
	img_tilde_opponent = torch.zeros((dim[0], dim[1], dim[2], dim[3]), device='cuda')
	img_tilde_opponent[:, 0:1, :, :] = nn.functional.conv2d(img_pad[:, 0:1, :, :], s_a.cuda(), padding=0)
	img_tilde_opponent[:, 1:2, :, :] = nn.functional.conv2d(img_pad[:, 1:2, :, :], s_rg.cuda(), padding=0)
	img_tilde_opponent[:, 2:3, :, :] = nn.functional.conv2d(img_pad[:, 2:3, :, :], s_by.cuda(), padding=0)
	img_tilde_linear_rgb = color_space_transform(img_tilde_opponent, 'ycxcz2linrgb')
	return torch.clamp(img_tilde_linear_rgb, 0.0, 1.0)


def hunt_adjustment(img):
	L = img[:, 0:1, :, :]
	img_h = torch.zeros(img.size(), device='cuda')
	img_h[:, 0:1, :, :] = L
	img_h[:, 1:2, :, :] = torch.mul((0.01 * L), img[:, 1:2, :, :])
	img_h[:, 2:3, :, :] = torch.mul((0.01 * L), img[:, 2:3, :, :])
	return img_h


def hyab(reference, test, eps):
	delta = reference - test
	root = torch.sqrt(torch.clamp(torch.pow(delta[:, 0:1, :, :], 2), min=eps))
	delta_norm = torch.norm(delta[:, 1:3, :, :], dim=1, keepdim=True)
	return root + delta_norm                                         


def redistribute_errors(power_deltaE_hyab, cmax, pc, pt):
	deltaE_c = torch.zeros(power_deltaE_hyab.size(), device='cuda')
	pccmax = pc * cmax
	deltaE_c = torch.where(power_deltaE_hyab < pccmax, (pt / pccmax) * power_deltaE_hyab, pt + ((power_deltaE_hyab - pccmax) / (cmax - pccmax)) * (1.0 - pt))
	return deltaE_c


def feature_detection(img_y, pixels_per_degree, feature_type):
	w = 0.082
	sd = 0.5 * w * pixels_per_degree
	radius = int(np.ceil(3 * sd))
	[x, y] = np.meshgrid(range(-radius, radius+1), range(-radius, radius+1))
	g = np.exp(-(x ** 2 + y ** 2) / (2 * sd * sd))
	if feature_type == 'edge':                 
		Gx = np.multiply(-x, g)
	else:                  
		Gx = np.multiply(x ** 2 / (sd * sd) - 1, g)
	negative_weights_sum = -np.sum(Gx[Gx < 0])
	positive_weights_sum = np.sum(Gx[Gx > 0])
	Gx = torch.Tensor(Gx)
	Gx = torch.where(Gx < 0, Gx / negative_weights_sum, Gx / positive_weights_sum)
	Gx = Gx.unsqueeze(0).unsqueeze(0).cuda()
	featuresX = nn.functional.conv2d(nn.functional.pad(img_y, (radius, radius, radius, radius), mode='replicate'), Gx, padding=0)
	featuresY = nn.functional.conv2d(nn.functional.pad(img_y, (radius, radius, radius, radius), mode='replicate'), torch.transpose(Gx, 2, 3), padding=0)
	return torch.cat((featuresX, featuresY), dim=1)


def color_space_transform(input_color, fromSpace2toSpace):
	dim = input_color.size()
	reference_illuminant = torch.tensor([[[0.950428545]], [[1.000000000]], [[1.088900371]]]).cuda()
	inv_reference_illuminant = torch.tensor([[[1.052156925]], [[1.000000000]], [[0.918357670]]]).cuda()
	if fromSpace2toSpace == "srgb2linrgb":
		limit = 0.04045
		transformed_color = torch.where(
			input_color > limit,
			torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4),
			input_color / 12.92
		)                               
	elif fromSpace2toSpace == "linrgb2srgb":
		limit = 0.0031308
		transformed_color = torch.where(
			input_color > limit,
			1.055 * torch.pow(torch.clamp(input_color, min=limit), (1.0 / 2.4)) - 0.055,
			12.92 * input_color
		)
	elif fromSpace2toSpace in ["linrgb2xyz", "xyz2linrgb"]:
		if fromSpace2toSpace == "linrgb2xyz":
			a11 = 10135552 / 24577794
			a12 = 8788810  / 24577794
			a13 = 4435075  / 24577794
			a21 = 2613072  / 12288897
			a22 = 8788810  / 12288897
			a23 = 887015   / 12288897
			a31 = 1425312  / 73733382
			a32 = 8788810  / 73733382
			a33 = 70074185 / 73733382
		else:
			a11 = 3.241003275
			a12 = -1.537398934
			a13 = -0.498615861
			a21 = -0.969224334
			a22 = 1.875930071
			a23 = 0.041554224
			a31 = 0.055639423
			a32 = -0.204011202
			a33 = 1.057148933
		A = torch.Tensor([[a11, a12, a13],
						  [a21, a22, a23],
						  [a31, a32, a33]])
		input_color = input_color.view(dim[0], dim[1], dim[2]*dim[3]).cuda()          
		transformed_color = torch.matmul(A.cuda(), input_color)
		transformed_color = transformed_color.view(dim[0], dim[1], dim[2], dim[3])
	elif fromSpace2toSpace == "xyz2ycxcz":
		input_color = torch.mul(input_color, inv_reference_illuminant)
		y = 116 * input_color[:, 1:2, :, :] - 16
		cx = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
		cz = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])
		transformed_color = torch.cat((y, cx, cz), 1)
	elif fromSpace2toSpace == "ycxcz2xyz":
		y = (input_color[:, 0:1, :, :] + 16) / 116
		cx = input_color[:, 1:2, :, :] / 500
		cz = input_color[:, 2:3, :, :] / 200
		x = y + cx
		z = y - cz
		transformed_color = torch.cat((x, y, z), 1)
		transformed_color = torch.mul(transformed_color, reference_illuminant)
	elif fromSpace2toSpace == "xyz2lab":
		input_color = torch.mul(input_color, inv_reference_illuminant)
		delta = 6 / 29
		delta_square = delta * delta
		delta_cube = delta * delta_square
		factor = 1 / (3 * delta_square)
		clamped_term = torch.pow(torch.clamp(input_color, min=delta_cube), 1.0 / 3.0).to(dtype=input_color.dtype)
		div = (factor * input_color + (4 / 29)).to(dtype=input_color.dtype)
		input_color = torch.where(input_color > delta_cube, clamped_term, div)                               
		L = 116 * input_color[:, 1:2, :, :] - 16
		a = 500 * (input_color[:, 0:1, :, :] - input_color[:, 1:2, :, :])
		b = 200 * (input_color[:, 1:2, :, :] - input_color[:, 2:3, :, :])
		transformed_color = torch.cat((L, a, b), 1)
	elif fromSpace2toSpace == "lab2xyz":
		y = (input_color[:, 0:1, :, :] + 16) / 116
		a = input_color[:, 1:2, :, :] / 500
		b = input_color[:, 2:3, :, :] / 200
		x = y + a
		z = y - b
		xyz = torch.cat((x, y, z), 1)
		delta = 6 / 29
		delta_square = delta * delta
		factor = 3 * delta_square
		xyz = torch.where(xyz > delta, torch.pow(xyz, 3), factor * (xyz - 4 / 29))
		transformed_color = torch.mul(xyz, reference_illuminant)
	elif fromSpace2toSpace == "srgb2xyz":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
	elif fromSpace2toSpace == "srgb2ycxcz":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
	elif fromSpace2toSpace == "linrgb2ycxcz":
		transformed_color = color_space_transform(input_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
	elif fromSpace2toSpace == "srgb2lab":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	elif fromSpace2toSpace == "linrgb2lab":
		transformed_color = color_space_transform(input_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	elif fromSpace2toSpace == "ycxcz2linrgb":
		transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
	elif fromSpace2toSpace == "lab2srgb":
		transformed_color = color_space_transform(input_color, 'lab2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
	elif fromSpace2toSpace == "ycxcz2lab":
		transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	else:
		sys.exit('Error: The color transform %s is not defined!' % fromSpace2toSpace)
	return transformed_color
