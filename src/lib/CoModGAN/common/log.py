import copy


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

##############
# cfg_holder #
##############


@singleton
class cfg_unique_holder(object):
    def __init__(self):
        self.cfg = None
        # this is use to track the main codes.
        self.code = set()

    def save_cfg(self, cfg):
        self.cfg = copy.deepcopy(cfg)

    def add_code(self, code):
        """
        A new main code is reached and
            its name is added.
        """
        self.code.add(code)

def print_log(*console_info):
    console_info = [str(i) for i in console_info]
    console_info = ' '.join(console_info)
    print(console_info)
    try:
        log_file = cfg_unique_holder().cfg.train.log_file
    except:
        try:
            log_file = cfg_unique_holder().cfg.eval.log_file
        except:
            return
    # TODO: potential bug on both have train and eval
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(console_info + '\n')
