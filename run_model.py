from config import Config
from utils.read_header import read_header

def run_model(model, header, recording):
    
    
    header_tmp = read_header(header,from_file=False)
    sampling_frequency, resolution, age, sex, snomed_codes = header_tmp
    

    a = fdgdfgdfgdf