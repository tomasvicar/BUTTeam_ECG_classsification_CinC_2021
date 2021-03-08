
from helper_code import find_challenge_files,load_header,load_recording,get_leads
import os
import sys
import numpy as np
import scipy as sp
import scipy.io
from shutil import rmtree
from multiprocessing import Pool


def extract_leads_wfdb_custom(src_path,dst_path,leads):
    
    reduced_leads = leads
    num_reduced_leads = len(reduced_leads)

    full_header_files, full_recording_files = find_challenge_files(src_path)
    
    if os.path.isdir(dst_path):
        rmtree(dst_path)
    
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    
    
    src_paths = [src_path for filename in full_header_files]
    dst_paths = [dst_path for filename in full_header_files]
    
    reduced_leadss = [reduced_leads for filename in full_header_files]
    num_reduced_leadss = [num_reduced_leads for filename in full_header_files]


    with Pool() as pool:
        pool.starmap(resave_one, zip(full_header_files,full_recording_files,src_paths,dst_paths,reduced_leadss,num_reduced_leadss))
    
    
    
    
    
def resave_one(full_header_file,full_recording_file,src_path,dst_path,reduced_leads,num_reduced_leads):
    
    # Load full header and recording.
    full_header = load_header(full_header_file)
    full_recording = load_recording(full_recording_file, 'val')

    # Get full lead names from header and check that are reduced leads are available.
    full_leads = get_leads(full_header)
    num_full_leads = len(full_leads)

    if np.shape(full_recording)[0] != num_full_leads:
        print('The signal file {} is malformed: the dimensions of the signal file are inconsistent with the header file {}.'.format(full_recording_file, full_header_file))
        sys.exit()

    unavailable_leads = [lead for lead in reduced_leads if lead not in full_leads]
    if unavailable_leads:
        print('The lead(s) {} are not available in the header file {}.'.format(', '.join(unavailable_leads), full_header_file))
        sys.exit()

    # Create the reduced lead header and recording files.
    head, tail = os.path.split(full_header_file)
    reduced_header_file = os.path.join(dst_path, tail)

    head, tail = os.path.split(full_recording_file)
    reduced_recording_file = os.path.join(dst_path, tail)

    root, extension = os.path.splitext(tail)
    recording_id = root

    # Initialize outputs.
    full_lines = full_header.split('\n')
    reduced_lines = list()

    # For the first line, we need to update the recording number and the number of leads.
    entries = full_lines[0].split()
    entries[0] = recording_id
    entries[1] = str(num_reduced_leads)
    reduced_lines.append(' '.join(entries))

    # For the next lines that describe the leads, we need to update the signal filename.
    reduced_indices = list()
    for lead in reduced_leads:
        i = full_leads.index(lead)
        reduced_indices.append(i)
        entries = full_lines[i+1].split()
        entries[0] = recording_id
        reduced_lines.append(' '.join(entries))

    # For the remaining lines that describe the other data, we do not need to update anything.
    for i in range(num_full_leads+1, len(full_lines)):
        entries = full_lines[i].split()
        reduced_lines.append(' '.join(entries))

    # Save the reduced lead header and recording files.
    with open(reduced_header_file, 'w') as f:
        f.write('\n'.join(reduced_lines))

    reduced_recording = full_recording[reduced_indices, :]
    d = {'val': reduced_recording}
    sp.io.savemat(reduced_recording_file, d)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




