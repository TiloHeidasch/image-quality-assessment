
import os
import glob
import json
import argparse
from utils.utils import calc_mean_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def main(base_model_name, weights_file1, weights_file2, image_source, predictions_file, img_format='jpg'):
    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    # build model and load weights
    nima1 = Nima(base_model_name, weights=None)
    nima1.build()
    nima1.nima_model.load_weights(weights_file1)

    # initialize data generator
    data_generator1 = TestDataGenerator(samples, image_dir, 64, 10, nima1.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    predictions1 = predict(nima1.nima_model, data_generator1)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction1'] = calc_mean_score(predictions1[i])

    print(json.dumps(sorted(samples, key=lambda sample: sample['mean_score_prediction1']), indent=2))

    # build model and load weights
    nima2 = Nima(base_model_name, weights=None)
    nima2.build()
    nima2.nima_model.load_weights(weights_file2)

    # initialize data generator
    data_generator2 = TestDataGenerator(samples, image_dir, 64, 10, nima2.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    predictions2 = predict(nima2.nima_model, data_generator2)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction2'] = calc_mean_score(predictions2[i])

    print(json.dumps(sorted(samples, key=lambda sample: sample['mean_score_prediction2']), indent=2))
    
    for i, sample in enumerate(samples):
        sample['mean_score_sum'] = sample['mean_score_prediction1']+sample['mean_score_prediction2']
    
    print(json.dumps(sorted(samples, key=lambda sample: sample['mean_score_sum']), indent=2))

    if predictions_file is not None:
        save_json(samples, predictions_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w1', '--weights-file1', help='path of weights file', required=True)
    parser.add_argument('-w2', '--weights-file2', help='path of weights file', required=True)
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)

    args = parser.parse_args()

    main(**args.__dict__)
