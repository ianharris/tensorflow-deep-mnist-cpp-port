load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "mnist",
    srcs = ["mnist.cc",
        "image.cc","image.h",
        "label.cc","label.h",
        "utils.cc","utils.h"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ],
)
