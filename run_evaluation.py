from config import Config
from evaluate_model import evaluate_model
import json
from datetime import datetime
import sys



if __name__ == '__main__':
    
    if len(sys.argv)>1:
        Config.RESULTS_PATH = sys.argv[1]
        Config.MODELS_SEED = int(sys.argv[2])
    
    
    results = dict()
    
    for leads in Config.LEAD_LISTS:
        
        
        output_directory = Config.DATA_RESAVE_PATH + '/output' + ', '.join(leads).replace('(','').replace(')','').replace(' ','').replace(',','_').replace('\'','')

        data_directory = Config.DATA_RESAVE_PATH  + '/test_data' + ', '.join(leads).replace('(','').replace(')','').replace(' ','').replace(',','_').replace('\'','')

        classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_measure_classes, challenge_metric = evaluate_model(data_directory, output_directory)
    
        print(str(len(leads)) )
        print(challenge_metric)
    
        results[str(len(leads))  + '-' + ','.join(leads)] = challenge_metric
        
        
    with open(Config.RESULTS_PATH + '/results_' + datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + '.json', 'w') as outfile:
        json.dump(results, outfile)