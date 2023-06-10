from argparse import ArgumentParser
from os import unlink
from .anns.faiss import FaissIndexUnderTestFactory
from .runner import Runner


def create_random_dataset(count, metric):
    from .dataset import RandomDataset
    dataset = RandomDataset(
        f'random_dataset_{count}_{metric}.h5', dimension=256, count=count, normalize=True, metric=metric)
    return dataset


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='config.yml', help='config file')
    args = parser.parse_args()
    index_factory = FaissIndexUnderTestFactory()

    for count in (1000000, 2000000, 5000000):
        for metric in ('ip', 'l2'):
            for nq in (1, 10, 100, 200):
                for topk in (1, 10, 100):
                    runner = Runner(create_random_dataset(count, metric), index_factory,
                                    dimension=256, gpu=True, metric_type=metric, nq=nq, topk=topk)
                    runner.run()
            unlink(runner.dataset.hd5_file.filename)


if __name__ == '__main__':
    main()
