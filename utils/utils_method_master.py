def get_algorithm_denoiser (method):
    # You should add denoiser name and algorithm name for all methods here
    
    method_data = {
        'A-Proposed':       {'algorithm': 'PnP-PDS', 'denoiser' : 'DnCNN'},
        'A-PnPFBS-DnCNN' :  {'algorithm' : 'PnP-FBS', 'denoiser' : 'DnCNN'},
        'A-PnPPDS-BM3D'   : {'algorithm' : 'PnP-PDS', 'denoiser' : 'BM3D'},
        'A-PnPFBS-BM3D' :   {'algorithm' : 'PnP-FBS', 'denoiser' : 'BM3D'},
        'A-PDS-TV' :        {'algorithm' : 'PDS', 'denoiser' : ''},
        'A-RED-DnCNN' :     {'algorithm' : 'RED-SD', 'denoiser' : 'DnCNN'},
        'A-PnPPDS-unstable-DnCNN' : {'algorithm' : 'PnP-PDS', 'denoiser' : 'DnCNN (unstable)'},

        'B-Proposed':       {'algorithm': 'PnP-PDS', 'denoiser' : 'DnCNN'},

        'C-Proposed':       {'algorithm': 'PnP-PDS', 'denoiser' : 'DnCNN'},
        'C-PnPPDS-BM3D' :   {'algorithm': 'PnP-PDS', 'denoiser' : 'BM3D'},
        'C-PnPADMM-DnCNN':  {'algorithm': 'PnP-ADMM', 'denoiser' : 'DnCNN'},
        'C-RED-DnCNN':      {'algorithm': 'RED-ADMM', 'denoiser' : 'DnCNN'},
        'C-PnP-unstable-DnCNN': {'algorithm': 'PnP-PDS', 'denoiser' : 'DnCNN (unstable)'},

    }    

    if (method in method_data):
        algorithm = method_data[method]['algorithm']
        denoiser  = method_data[method]['denoiser']
    else :
        algorithm = 'unknown algorithm'
        denoiser  = 'unknown denoiser'

    return algorithm, denoiser