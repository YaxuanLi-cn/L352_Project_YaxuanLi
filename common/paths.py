import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CODE = os.path.join(ROOT, "code")
COMMON = os.path.join(CODE, "common")
DATA = os.path.join(ROOT, "data")
DATASETS = os.path.join(DATA, "datasets")
RECORDS = os.path.join(ROOT, "records")
DOCS = os.path.join(ROOT, "docs")
E1 = "e1_degradation_ladders"                                           
E2 = "e2_observer_agreement"                                              
E3 = "e3_video_attack"                                
E4 = "e4_self_overrating"                                  
E5 = "e5_image_attack"                                  
E6 = "e6_user_study"                                       
E7 = "e7_uncertainty_strata"                           
E8 = "e8_drift"                                          
S1 = "s1_exploration_truck"                                           
S2 = "s2_sufficiency_matrix"                                        
S3 = "s3_tint_truck"                                          
S4 = "s4_tint_scenes"                               
S5 = "s5_detectors"                                          
S6 = "s6_round2"                                            


def code(exp, *parts):
    return os.path.join(CODE, "experiments", exp, *parts)


def data(exp, *parts):
    return os.path.join(DATA, "experiments", exp, *parts)


def rec(exp, *parts):
    return os.path.join(RECORDS, exp, *parts)


def rel(path):
    return os.path.relpath(path, ROOT)
PART1_RENDERS = data(E1)                                                                                                          
PART1_RESULTS = rec(E1, "results")                                                                                              
GSQA_RESULTS = rec(E2, "gsqa")                                                                                                        
NVSQA_RESULTS = rec(E2, "nvsqa")                                  
UNANIMITY = rec(E2, "unanimity")                                                                               


def subjective_results(tag):
    return rec(E2, tag)
FLICKER_RESULTS = rec(E3, "results")                                                                  
STAGE_A_CKPT = data(E4, "stage_a_ckpt")                                                          
ORIG = data(E4, "original_run")                                                       
ORIG_RUNS = os.path.join(ORIG, "runs")                       
ORIG_STIMULI = os.path.join(ORIG, "stimuli")                 
ORIG_RESULTS = rec(E4, "original_run", "results")                                                      
ORIG_TABLES = os.path.join(ORIG_RESULTS, "tables")                  
RERUN = data(E4, "rerun")                                                                
RERUN_RUNS = os.path.join(RERUN, "runs")                   
RERUN_CKPT = os.path.join(RERUN, "ckpt")                                
RERUN_STIMULI = os.path.join(RERUN, "stimuli")                                                                                        
RERUN_VIEWS = os.path.join(RERUN, "views")                                      
RERUN_WHOLE = rec(E4, "rerun", "whole")                                                             
RERUN_RESULTS = os.path.join(RERUN_WHOLE, "results")             
RERUN_TABLES = os.path.join(RERUN_RESULTS, "tables")              
RERUN_COMPARE = rec(E4, "rerun", "compare")               
SEED_CKPT = data(E7, "ckpt_seeds")                                     
PILOT = rec(E7, "pilot")                                                                                 
PILOT_LOGS = rec(E7, "logs")                                                        
DRIFT = rec(E8, "two_trainings")                                                        
DECOMPOSITION = rec(E8, "decomposition")                             
S1_RUNS = data(E5, S1, "runs")                                 
S2_RUNS = data(E5, S2, "runs")                                 
S2_AUX = data(E5, S2, "aux")                                                      
S2_REC = rec(E5, S2)                                                                                   
THRESHOLDS = os.path.join(S2_REC, "thresholds.csv")                              
S3_RUNS = data(E5, S3, "runs")                                 
S3_REC = rec(E5, S3)                                     
S4_RUNS = data(E5, S4, "runs")                                 
S4_REC = rec(E5, S4)                                                                                
S5_REC = rec(E5, S5)                                                       
S6_RUNS = data(E5, S6, "runs")                                 
S6_REC = rec(E5, S6)                                                                  
STUDY_DATA = data(E6, "study_data")                                                                                         
STUDY_RESULTS = rec(E6, "results")                                             
REPORT = os.path.join(DOCS, "report")                  
REPORT_TOOLS = os.path.join(REPORT, "tools")              
SLIDES = os.path.join(DOCS, "slides")                 
PAPER8_PDF = os.path.join(DOCS, "paper_8page", "ICCV_8page_draft.pdf")                               
LEGACY = [
    ("goodhart/ckpt", STAGE_A_CKPT),                                
    ("goodhart/runs", ORIG_RUNS),                           
    ("goodhart/stimuli", ORIG_STIMULI),                       
    ("goodhart/results", ORIG_RESULTS),                    
    ("uncertainty_pilot/out/rerun/ckpt", RERUN_CKPT),                             
    ("uncertainty_pilot/out/rerun/runs", RERUN_RUNS),                      
    ("uncertainty_pilot/out/rerun/stimuli", RERUN_STIMULI),               
    ("uncertainty_pilot/out/views", RERUN_VIEWS),                    
    ("uncertainty_pilot/out/ckpt_seeds", SEED_CKPT),                     
    ("uncertainty_pilot/out/pilot", PILOT),                            
    ("outputs/part1", PART1_RENDERS),                   
    ("results/part1", PART1_RESULTS),                  
    ("dataset", DATASETS),                         
    ("innovation/ft/runs", S1_RUNS),                    
    ("innovation/c8_sufficiency/runs", S2_RUNS),                             
    ("innovation/c8_sufficiency/aux", S2_AUX),                                
    ("innovation/c11_tint/runs", S3_RUNS),              
    ("innovation/c8_colour/runs", S4_RUNS),             
    ("paper_iccv/analysis/ext/runs", S6_RUNS),                           
    ("paper_iccv/study/data", STUDY_DATA),              
]


def legacy(path):
    p = path
    for prefix in ("/root/rafel_project/",):
        if p.startswith(prefix):
            p = p[len(prefix):]
    if os.path.isabs(p):
        return p
    for old, new in LEGACY:
        if p == old or p.startswith(old + "/"):
            return os.path.normpath(os.path.join(new, os.path.relpath(p, old)))
    return os.path.join(ROOT, p)
