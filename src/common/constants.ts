// Copyright (C) 2021, Mindee.

// This program is licensed under the Apache License version 2.
// See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

// code cfg
export const CODE_LEN = 16;
export const INTERVAL = 200;
export const MIN_RATIO = 0.4;

// Detection cfg

export const DET_MEAN = 0.785;
export const DET_STD = 0.275;

export const DET_CONFIG = {
  db_mobilenet_v2: {
    value: "db_mobilenet_v2",
    label: "DB (MobileNet V2)",
    path: "models/db_mobilenet_v2/model.json",
    // value: "detector_model",
    // label: "detector_model",
    // path: "models/detector_tfjs_model/model.json",
    height: 512,
    width: 512,
  },
};

// Recognition cfg

export const REC_MEAN = 0.694;
export const REC_STD = 0.298;

export const RECO_CONFIG = {
  crnn_vgg16_bn: {
    value: "crnn_mobilenet_v2",
    label: "CRNN (MobileNet V2)",
    path: "models/crnn_mobilenet_v2/model.json",
    // value: "recognizer_model",
    // label: "recognizer_model",
    // path: "models/recognizer_tfjs_model/model.json",
    height: 32,
    width: 128,
  },
};

export const VOCAB =
  "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
