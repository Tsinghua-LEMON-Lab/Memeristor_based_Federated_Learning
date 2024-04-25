# Memersitor_based_Federated_Learning
The code for paper "Federated learning using a memristor compute-in-memory chip with in-situ PUF and TRNG"

This code contains a self-built BFV-based homomorphic encryption library, an RRNS library, and the sepsis detection task accomplishes 4-user federated learning by calling these two libraries.

The main function is FL_MKBFV_MIMIC\src\project\train_sepsis_fl_w_enc
The self-built homomorphic encryption library is FL_MKBFV_MIMIC\src\project\enclib_forFL, which calls enclib_BASIC, enclib_CORE
RRNS library is FL_MKBFV_MIMIC\src\project\RNSlib
LSTM Sepsis Network Model is in FL_MKBFV_MIMIC\src\project\mymodels
