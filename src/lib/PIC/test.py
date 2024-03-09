from Pluralistic.options import test_options
from Pluralistic.dataloader import data_loader
from PIC.model import create_model
from Pluralistic.util import visualizer
from itertools import islice
import yaml
from PEPSI.utils.io import YamlHandler

if __name__=='__main__':
    # get testing options
    # yamlHandler = YamlHandler('./PL.yaml')
    # opt = test_options.TestOptions().parse()
    # argsDict = {argName: getattr(opt, argName) for argName in vars(opt)}
    # yamlHandler.write_yaml(opt)
    from omegaconf import OmegaConf
    with open('./PL.yaml', 'r') as f:
        opt = OmegaConf.create(yaml.safe_load(f))
    # creat a dataset
    dataset = data_loader.dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('testing images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    model.eval()
    # create a visualizer
    # visualizer = visualizer.Visualizer(opt)

    for i, data in enumerate(islice(dataset, opt.how_many)):
        model.set_input(data)
        model.test()
