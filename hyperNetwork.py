import meteo_api as API




class HypNet:
    """This class create a hypernetwork transformers based
    that output weights for a PhysNet to make direct prediction"""
    def __init__(
            self
    ) ->None:
        
        self.name = None