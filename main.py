from utils import Utils
from models import Models


if __name__ == '__main__':

    utils = Utils()
    models = Models()

    dataset = utils.load_from_csv('in/happiness.csv')
    print(dataset.head(5))

    x, y = utils.get_features_target(dataset, ['score', 'rank', 'country'],
                                     ['score'])

    models.grid_training(x, y)
