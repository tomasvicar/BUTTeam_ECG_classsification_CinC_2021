


from config import Config
from evaluate_model import evaluate_model



if __name__ == '__main__':
    
    
    
    
    for leads in Config.LEAD_LISTS:
        
        
        output_directory = Config.DATA_RESAVE_PATH + '/output' + str(len(leads))  

        data_directory = Config.DATA_RESAVE_PATH  + '/test_data' + str(len(leads))  

        classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_measure_classes, challenge_metric = evaluate_model(data_directory, output_directory)
    
        print(str(len(leads)) )
        print(challenge_metric )
    