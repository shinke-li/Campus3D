import logging
import logging.config
import os.path as osp
import yaml

CFG_FILE = osp.join(osp.dirname(__file__), 'logger.yaml')


def setup_logging(default_path=CFG_FILE, default_level=logging.INFO ):
    path = default_path
    if osp.exists(osp.abspath(path)):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
            return __get_collect_logger(config)
    else:
        logging.basicConfig(level=default_level)
        logging.warning('Config file \'{}\' cannot be found.'.format(path))
        return None, None


def __get_collect_logger(config, name='collects'):
    collect_logger = logging.getLogger(name)
    return collect_logger, \
           {n: hdl for n, hdl in
            zip(config['handlers'].keys(), collect_logger.handlers)}


__COLLECT_LOGGER, __COLLECT_HANDLERS = setup_logging()
LOG = logging.getLogger()

def gen_logger(name, file=None, copy_root=True, propagate=False, ):
    logger = logging.getLogger(name)
    logger.propagate = propagate
    if file is not None:
        __add_file_handler(logger, file)
    if copy_root:
        for hdl in LOG.handlers:
            logger.addHandler(hdl)
    return logger

def file_output(file, level=None):
    if __COLLECT_HANDLERS is not None:
        __add_file_handler(LOG, file)
    else:
        LOG.addHandler(logging.FileHandler(file, mode='a'))
    if level is not None:
        level = getattr(logging, level.upper())
        LOG.handlers[-1].setLevel(level)


def __add_file_handler(logger, file_name):
    fh = logging.FileHandler(file_name, mode='a')
    fh.setFormatter(__COLLECT_HANDLERS['file'].formatter)
    logger.addHandler(fh)

if __name__ == '__main__':
    LOG.error('asdasd')
    f_log = gen_logger('train', file='train.log')
    f_log.error('hello')