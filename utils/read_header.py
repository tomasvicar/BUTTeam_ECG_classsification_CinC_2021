import numpy as np
import math

from config import Config

def read_header(file_name,from_file=True,remap=False):
    """Function saves information about the patient from header file in this order:
    sampling frequency, length of the signal, voltage resolution, age, sex, list of diagnostic labels both in
    SNOMED and abbreviations (sepate lists)"""

    sampling_frequency, resolution, age, sex, snomed_codes = [], [], [], [], []

    def string_to_float(input_string):
        """Converts string to floating point number"""
        try:
            value = float(input_string)
        except ValueError:
            value = None

        if math.isnan(value):
            return None
        else:
            return value


    if from_file:
        lines=[]
        with open(file_name, "r") as file:
            for line_idx, line in enumerate(file):
                lines.append(line)
    else:
        lines=file_name.split('\n')

    # Read line 15 in header file and parse string with labels

    name = lines[0].split(" ")[0]
    
    snomed_codes = []
    resolution = []
    leads = []

    for line_idx, line in enumerate(lines):
        if line_idx == 0:
            sampling_frequency = float(line.split(" ")[2])
            continue
        if 1<=line_idx and line.startswith(name):
            resolution.append(string_to_float(line.split(" ")[2].replace("/mV", "").replace("/mv", "")))
            leads.append(line.split(" ")[-1])
            
            continue
        if line.startswith('#Age'):
            age = string_to_float(line.replace("#Age:","").replace("#Age","").rstrip("\n").strip())
            continue
        if line.startswith('#Sex'):
            sex = line.replace("#Sex:","").replace("#Sex","").rstrip("\n").strip().lower()
            
            continue
        if line.startswith('#Dx'):
            if from_file:
                snomed_codes = line.replace("#Dx:","").replace("#Dx","").rstrip("\n").strip().split(",")
                
                snomed_codes = [Config.SNOMED2IDX_MAP.get(item, item) for item in snomed_codes]
            continue

    return sampling_frequency, resolution, age, sex, snomed_codes, leads