static int kernel1_I_shape_0[] = {32, 512, 64, 1};
static int kernel1_F_shape_0[] = {32, 512, 64, 1};
static int kernel1_O_shape_0[] = {32, 512, 512, 1};
static int kernel1_I_shape_1[] = {32, 512, 512, 1};
static int kernel1_F_shape_1[] = {32, 64, 512, 1};
static int kernel1_O_shape_1[] = {32, 512, 64, 1};
static int kernel1_I_shape_2[] = {32, 256, 80, 1};
static int kernel1_F_shape_2[] = {32, 256, 80, 1};
static int kernel1_O_shape_2[] = {32, 256, 256, 1};
static int kernel1_I_shape_3[] = {32, 256, 256, 1};
static int kernel1_F_shape_3[] = {32, 80, 256, 1};
static int kernel1_O_shape_3[] = {32, 256, 80, 1};
static int kernel1_I_shape_4[] = {32, 208, 80, 1};
static int kernel1_F_shape_4[] = {32, 208, 80, 1};
static int kernel1_O_shape_4[] = {32, 208, 208, 1};
static int kernel1_I_shape_5[] = {32, 208, 208, 1};
static int kernel1_F_shape_5[] = {32, 80, 208, 1};
static int kernel1_O_shape_5[] = {32, 208, 80, 1};
static int kernel1_I_shape_6[] = {32, 1024, 64, 1};
static int kernel1_F_shape_6[] = {32, 512, 64, 1};
static int kernel1_O_shape_6[] = {32, 1024, 512, 1};
static int kernel1_I_shape_7[] = {32, 1024, 512, 1};
static int kernel1_F_shape_7[] = {32, 64, 512, 1};
static int kernel1_O_shape_7[] = {32, 1024, 64, 1};
//------------------------------------------------------------------------------------------
static int kernel2_I_shape_0[] = {4, 64, 112, 112};
static int kernel2_F_shape_0[] = {192, 64, 3, 3};
static int kernel2_O_shape_0[] = {4, 192, 55, 55};
static int kernel2_I_shape_1[] = {4, 32, 147, 147};
static int kernel2_F_shape_1[] = {64, 32, 3, 3};
static int kernel2_O_shape_1[] = {4, 64, 73, 73};
static int kernel2_I_shape_2[] = {4, 64, 56, 56};
static int kernel2_F_shape_2[] = {128, 64, 3, 3};
static int kernel2_O_shape_2[] = {4, 128, 54, 54};
static int kernel2_I_shape_3[] = {4, 128, 28, 28};
static int kernel2_F_shape_3[] = {256, 128, 3, 3};
static int kernel2_O_shape_3[] = {4, 256, 26, 26};
static int kernel2_I_shape_4[] = {4, 16, 227, 227};
static int kernel2_F_shape_4[] = {64, 16, 3, 3};
static int kernel2_O_shape_4[] = {4, 64, 57, 57};
static int kernel2_I_shape_5[] = {4, 64, 56, 56};
static int kernel2_F_shape_5[] = {64, 64, 1, 1};
static int kernel2_O_shape_5[] = {4, 64, 56, 56};
static int kernel2_I_shape_6[] = {4, 64, 56, 56};
static int kernel2_F_shape_6[] = {64, 64, 1, 1};
static int kernel2_O_shape_6[] = {4, 64, 56, 56};
static int kernel2_I_shape_7[] = {4, 256, 56, 56};
static int kernel2_F_shape_7[] = {256, 256, 1, 1};
static int kernel2_O_shape_7[] = {4, 256, 56, 56};
//------------------------------------------------------------------------------------------
static unsigned int kernel1_gridDim_0[] = {8, 8, 32};
static unsigned int kernel1_blockDim_0[] = {8, 8, 1};
static unsigned int kernel1_gridDim_1[] = {1, 8, 32};
static unsigned int kernel1_blockDim_1[] = {8, 8, 1};
static unsigned int kernel1_gridDim_2[] = {4, 4, 32};
static unsigned int kernel1_blockDim_2[] = {8, 8, 1};
static unsigned int kernel1_gridDim_3[] = {5, 4, 32};
static unsigned int kernel1_blockDim_3[] = {8, 8, 1};
static unsigned int kernel1_gridDim_4[] = {13, 13, 32};
static unsigned int kernel1_blockDim_4[] = {8, 8, 1};
static unsigned int kernel1_gridDim_5[] = {5, 13, 32};
static unsigned int kernel1_blockDim_5[] = {8, 8, 1};
static unsigned int kernel1_gridDim_6[] = {8, 16, 32};
static unsigned int kernel1_blockDim_6[] = {8, 8, 1};
static unsigned int kernel1_gridDim_7[] = {1, 16, 32};
static unsigned int kernel1_blockDim_7[] = {8, 8, 1};
//------------------------------------------------------------------------------------------
static unsigned int kernel2_gridDim_0[] = {55, 11, 12};
static unsigned int kernel2_blockDim_0[] = {1, 1, 16};
static unsigned int kernel2_gridDim_1[] = {73, 73, 8};
static unsigned int kernel2_blockDim_1[] = {1, 1, 8};
static unsigned int kernel2_gridDim_2[] = {2, 9, 8};
static unsigned int kernel2_blockDim_2[] = {27, 1, 8};
static unsigned int kernel2_gridDim_3[] = {2, 13, 16};
static unsigned int kernel2_blockDim_3[] = {13, 1, 8};
static unsigned int kernel2_gridDim_4[] = {3, 57, 8};
static unsigned int kernel2_blockDim_4[] = {19, 1, 8};
static unsigned int kernel2_gridDim_5[] = {1, 28, 4};
static unsigned int kernel2_blockDim_5[] = {56, 1, 2};
static unsigned int kernel2_gridDim_6[] = {1, 28, 4};
static unsigned int kernel2_blockDim_6[] = {56, 1, 2};
static unsigned int kernel2_gridDim_7[] = {2, 28, 16};
static unsigned int kernel2_blockDim_7[] = {14, 1, 16};
//------------------------------------------------------------------------------------------
static unsigned int hfuse_gridDim_0[] = {7260, 1, 1};
static unsigned int hfuse_blockDim_0[] = {96, 1, 1};
static unsigned int hfuse_gridDim_1[] = {42632, 1, 1};
static unsigned int hfuse_blockDim_1[] = {96, 1, 1};
static unsigned int hfuse_gridDim_2[] = {2048, 1, 1};
static unsigned int hfuse_blockDim_2[] = {288, 1, 1};
static unsigned int hfuse_gridDim_3[] = {2048, 1, 1};
static unsigned int hfuse_blockDim_3[] = {192, 1, 1};
static unsigned int hfuse_gridDim_4[] = {2048, 1, 1};
static unsigned int hfuse_blockDim_4[] = {224, 1, 1};
static unsigned int hfuse_gridDim_5[] = {2048, 1, 1};
static unsigned int hfuse_blockDim_5[] = {192, 1, 1};
static unsigned int hfuse_gridDim_6[] = {2048, 1, 1};
static unsigned int hfuse_blockDim_6[] = {192, 1, 1};
static unsigned int hfuse_gridDim_7[] = {2048, 1, 1};
static unsigned int hfuse_blockDim_7[] = {288, 1, 1};
static unsigned int hfuse_gridDim_8[] = {7260, 1, 1};
static unsigned int hfuse_blockDim_8[] = {96, 1, 1};
static unsigned int hfuse_gridDim_9[] = {42632, 1, 1};
static unsigned int hfuse_blockDim_9[] = {96, 1, 1};
static unsigned int hfuse_gridDim_10[] = {256, 1, 1};
static unsigned int hfuse_blockDim_10[] = {288, 1, 1};
static unsigned int hfuse_gridDim_11[] = {416, 1, 1};
static unsigned int hfuse_blockDim_11[] = {192, 1, 1};
static unsigned int hfuse_gridDim_12[] = {1368, 1, 1};
static unsigned int hfuse_blockDim_12[] = {224, 1, 1};
static unsigned int hfuse_gridDim_13[] = {256, 1, 1};
static unsigned int hfuse_blockDim_13[] = {192, 1, 1};
static unsigned int hfuse_gridDim_14[] = {256, 1, 1};
static unsigned int hfuse_blockDim_14[] = {192, 1, 1};
static unsigned int hfuse_gridDim_15[] = {896, 1, 1};
static unsigned int hfuse_blockDim_15[] = {288, 1, 1};
static unsigned int hfuse_gridDim_16[] = {7260, 1, 1};
static unsigned int hfuse_blockDim_16[] = {96, 1, 1};
static unsigned int hfuse_gridDim_17[] = {42632, 1, 1};
static unsigned int hfuse_blockDim_17[] = {96, 1, 1};
static unsigned int hfuse_gridDim_18[] = {512, 1, 1};
static unsigned int hfuse_blockDim_18[] = {288, 1, 1};
static unsigned int hfuse_gridDim_19[] = {512, 1, 1};
static unsigned int hfuse_blockDim_19[] = {192, 1, 1};
static unsigned int hfuse_gridDim_20[] = {1368, 1, 1};
static unsigned int hfuse_blockDim_20[] = {224, 1, 1};
static unsigned int hfuse_gridDim_21[] = {512, 1, 1};
static unsigned int hfuse_blockDim_21[] = {192, 1, 1};
static unsigned int hfuse_gridDim_22[] = {512, 1, 1};
static unsigned int hfuse_blockDim_22[] = {192, 1, 1};
static unsigned int hfuse_gridDim_23[] = {896, 1, 1};
static unsigned int hfuse_blockDim_23[] = {288, 1, 1};
static unsigned int hfuse_gridDim_24[] = {7260, 1, 1};
static unsigned int hfuse_blockDim_24[] = {96, 1, 1};
static unsigned int hfuse_gridDim_25[] = {42632, 1, 1};
static unsigned int hfuse_blockDim_25[] = {96, 1, 1};
static unsigned int hfuse_gridDim_26[] = {640, 1, 1};
static unsigned int hfuse_blockDim_26[] = {288, 1, 1};
static unsigned int hfuse_gridDim_27[] = {640, 1, 1};
static unsigned int hfuse_blockDim_27[] = {192, 1, 1};
static unsigned int hfuse_gridDim_28[] = {1368, 1, 1};
static unsigned int hfuse_blockDim_28[] = {224, 1, 1};
static unsigned int hfuse_gridDim_29[] = {640, 1, 1};
static unsigned int hfuse_blockDim_29[] = {192, 1, 1};
static unsigned int hfuse_gridDim_30[] = {640, 1, 1};
static unsigned int hfuse_blockDim_30[] = {192, 1, 1};
static unsigned int hfuse_gridDim_31[] = {896, 1, 1};
static unsigned int hfuse_blockDim_31[] = {288, 1, 1};
static unsigned int hfuse_gridDim_32[] = {7260, 1, 1};
static unsigned int hfuse_blockDim_32[] = {96, 1, 1};
static unsigned int hfuse_gridDim_33[] = {42632, 1, 1};
static unsigned int hfuse_blockDim_33[] = {96, 1, 1};
static unsigned int hfuse_gridDim_34[] = {5408, 1, 1};
static unsigned int hfuse_blockDim_34[] = {288, 1, 1};
static unsigned int hfuse_gridDim_35[] = {5408, 1, 1};
static unsigned int hfuse_blockDim_35[] = {192, 1, 1};
static unsigned int hfuse_gridDim_36[] = {5408, 1, 1};
static unsigned int hfuse_blockDim_36[] = {224, 1, 1};
static unsigned int hfuse_gridDim_37[] = {5408, 1, 1};
static unsigned int hfuse_blockDim_37[] = {192, 1, 1};
static unsigned int hfuse_gridDim_38[] = {5408, 1, 1};
static unsigned int hfuse_blockDim_38[] = {192, 1, 1};
static unsigned int hfuse_gridDim_39[] = {5408, 1, 1};
static unsigned int hfuse_blockDim_39[] = {288, 1, 1};
static unsigned int hfuse_gridDim_40[] = {7260, 1, 1};
static unsigned int hfuse_blockDim_40[] = {96, 1, 1};
static unsigned int hfuse_gridDim_41[] = {42632, 1, 1};
static unsigned int hfuse_blockDim_41[] = {96, 1, 1};
static unsigned int hfuse_gridDim_42[] = {2080, 1, 1};
static unsigned int hfuse_blockDim_42[] = {288, 1, 1};
static unsigned int hfuse_gridDim_43[] = {2080, 1, 1};
static unsigned int hfuse_blockDim_43[] = {192, 1, 1};
static unsigned int hfuse_gridDim_44[] = {2080, 1, 1};
static unsigned int hfuse_blockDim_44[] = {224, 1, 1};
static unsigned int hfuse_gridDim_45[] = {2080, 1, 1};
static unsigned int hfuse_blockDim_45[] = {192, 1, 1};
static unsigned int hfuse_gridDim_46[] = {2080, 1, 1};
static unsigned int hfuse_blockDim_46[] = {192, 1, 1};
static unsigned int hfuse_gridDim_47[] = {2080, 1, 1};
static unsigned int hfuse_blockDim_47[] = {288, 1, 1};
static unsigned int hfuse_gridDim_48[] = {7260, 1, 1};
static unsigned int hfuse_blockDim_48[] = {96, 1, 1};
static unsigned int hfuse_gridDim_49[] = {42632, 1, 1};
static unsigned int hfuse_blockDim_49[] = {96, 1, 1};
static unsigned int hfuse_gridDim_50[] = {4096, 1, 1};
static unsigned int hfuse_blockDim_50[] = {288, 1, 1};
static unsigned int hfuse_gridDim_51[] = {4096, 1, 1};
static unsigned int hfuse_blockDim_51[] = {192, 1, 1};
static unsigned int hfuse_gridDim_52[] = {4096, 1, 1};
static unsigned int hfuse_blockDim_52[] = {224, 1, 1};
static unsigned int hfuse_gridDim_53[] = {4096, 1, 1};
static unsigned int hfuse_blockDim_53[] = {192, 1, 1};
static unsigned int hfuse_gridDim_54[] = {4096, 1, 1};
static unsigned int hfuse_blockDim_54[] = {192, 1, 1};
static unsigned int hfuse_gridDim_55[] = {4096, 1, 1};
static unsigned int hfuse_blockDim_55[] = {288, 1, 1};
static unsigned int hfuse_gridDim_56[] = {7260, 1, 1};
static unsigned int hfuse_blockDim_56[] = {96, 1, 1};
static unsigned int hfuse_gridDim_57[] = {42632, 1, 1};
static unsigned int hfuse_blockDim_57[] = {96, 1, 1};
static unsigned int hfuse_gridDim_58[] = {512, 1, 1};
static unsigned int hfuse_blockDim_58[] = {288, 1, 1};
static unsigned int hfuse_gridDim_59[] = {512, 1, 1};
static unsigned int hfuse_blockDim_59[] = {192, 1, 1};
static unsigned int hfuse_gridDim_60[] = {1368, 1, 1};
static unsigned int hfuse_blockDim_60[] = {224, 1, 1};
static unsigned int hfuse_gridDim_61[] = {512, 1, 1};
static unsigned int hfuse_blockDim_61[] = {192, 1, 1};
static unsigned int hfuse_gridDim_62[] = {512, 1, 1};
static unsigned int hfuse_blockDim_62[] = {192, 1, 1};
static unsigned int hfuse_gridDim_63[] = {896, 1, 1};
static unsigned int hfuse_blockDim_63[] = {288, 1, 1};
//------------------------------------------------------------------------------------------
static unsigned int bfuse_gridDim_0[] = {9308, 1, 1};
static unsigned int bfuse_blockDim_0[] = {64, 1, 1};
static unsigned int bfuse_gridDim_1[] = {44680, 1, 1};
static unsigned int bfuse_blockDim_1[] = {64, 1, 1};
static unsigned int bfuse_gridDim_2[] = {2192, 1, 1};
static unsigned int bfuse_blockDim_2[] = {216, 1, 1};
static unsigned int bfuse_gridDim_3[] = {2464, 1, 1};
static unsigned int bfuse_blockDim_3[] = {104, 1, 1};
static unsigned int bfuse_gridDim_4[] = {3416, 1, 1};
static unsigned int bfuse_blockDim_4[] = {152, 1, 1};
static unsigned int bfuse_gridDim_5[] = {2160, 1, 1};
static unsigned int bfuse_blockDim_5[] = {112, 1, 1};
static unsigned int bfuse_gridDim_6[] = {2160, 1, 1};
static unsigned int bfuse_blockDim_6[] = {112, 1, 1};
static unsigned int bfuse_gridDim_7[] = {2944, 1, 1};
static unsigned int bfuse_blockDim_7[] = {224, 1, 1};
static unsigned int bfuse_gridDim_8[] = {7516, 1, 1};
static unsigned int bfuse_blockDim_8[] = {64, 1, 1};
static unsigned int bfuse_gridDim_9[] = {42888, 1, 1};
static unsigned int bfuse_blockDim_9[] = {64, 1, 1};
static unsigned int bfuse_gridDim_10[] = {400, 1, 1};
static unsigned int bfuse_blockDim_10[] = {216, 1, 1};
static unsigned int bfuse_gridDim_11[] = {672, 1, 1};
static unsigned int bfuse_blockDim_11[] = {104, 1, 1};
static unsigned int bfuse_gridDim_12[] = {1624, 1, 1};
static unsigned int bfuse_blockDim_12[] = {152, 1, 1};
static unsigned int bfuse_gridDim_13[] = {368, 1, 1};
static unsigned int bfuse_blockDim_13[] = {112, 1, 1};
static unsigned int bfuse_gridDim_14[] = {368, 1, 1};
static unsigned int bfuse_blockDim_14[] = {112, 1, 1};
static unsigned int bfuse_gridDim_15[] = {1152, 1, 1};
static unsigned int bfuse_blockDim_15[] = {224, 1, 1};
static unsigned int bfuse_gridDim_16[] = {7772, 1, 1};
static unsigned int bfuse_blockDim_16[] = {64, 1, 1};
static unsigned int bfuse_gridDim_17[] = {43144, 1, 1};
static unsigned int bfuse_blockDim_17[] = {64, 1, 1};
static unsigned int bfuse_gridDim_18[] = {656, 1, 1};
static unsigned int bfuse_blockDim_18[] = {216, 1, 1};
static unsigned int bfuse_gridDim_19[] = {928, 1, 1};
static unsigned int bfuse_blockDim_19[] = {104, 1, 1};
static unsigned int bfuse_gridDim_20[] = {1880, 1, 1};
static unsigned int bfuse_blockDim_20[] = {152, 1, 1};
static unsigned int bfuse_gridDim_21[] = {624, 1, 1};
static unsigned int bfuse_blockDim_21[] = {112, 1, 1};
static unsigned int bfuse_gridDim_22[] = {624, 1, 1};
static unsigned int bfuse_blockDim_22[] = {112, 1, 1};
static unsigned int bfuse_gridDim_23[] = {1408, 1, 1};
static unsigned int bfuse_blockDim_23[] = {224, 1, 1};
static unsigned int bfuse_gridDim_24[] = {7900, 1, 1};
static unsigned int bfuse_blockDim_24[] = {64, 1, 1};
static unsigned int bfuse_gridDim_25[] = {43272, 1, 1};
static unsigned int bfuse_blockDim_25[] = {64, 1, 1};
static unsigned int bfuse_gridDim_26[] = {784, 1, 1};
static unsigned int bfuse_blockDim_26[] = {216, 1, 1};
static unsigned int bfuse_gridDim_27[] = {1056, 1, 1};
static unsigned int bfuse_blockDim_27[] = {104, 1, 1};
static unsigned int bfuse_gridDim_28[] = {2008, 1, 1};
static unsigned int bfuse_blockDim_28[] = {152, 1, 1};
static unsigned int bfuse_gridDim_29[] = {752, 1, 1};
static unsigned int bfuse_blockDim_29[] = {112, 1, 1};
static unsigned int bfuse_gridDim_30[] = {752, 1, 1};
static unsigned int bfuse_blockDim_30[] = {112, 1, 1};
static unsigned int bfuse_gridDim_31[] = {1536, 1, 1};
static unsigned int bfuse_blockDim_31[] = {224, 1, 1};
static unsigned int bfuse_gridDim_32[] = {12668, 1, 1};
static unsigned int bfuse_blockDim_32[] = {64, 1, 1};
static unsigned int bfuse_gridDim_33[] = {48040, 1, 1};
static unsigned int bfuse_blockDim_33[] = {64, 1, 1};
static unsigned int bfuse_gridDim_34[] = {5552, 1, 1};
static unsigned int bfuse_blockDim_34[] = {216, 1, 1};
static unsigned int bfuse_gridDim_35[] = {5824, 1, 1};
static unsigned int bfuse_blockDim_35[] = {104, 1, 1};
static unsigned int bfuse_gridDim_36[] = {6776, 1, 1};
static unsigned int bfuse_blockDim_36[] = {152, 1, 1};
static unsigned int bfuse_gridDim_37[] = {5520, 1, 1};
static unsigned int bfuse_blockDim_37[] = {112, 1, 1};
static unsigned int bfuse_gridDim_38[] = {5520, 1, 1};
static unsigned int bfuse_blockDim_38[] = {112, 1, 1};
static unsigned int bfuse_gridDim_39[] = {6304, 1, 1};
static unsigned int bfuse_blockDim_39[] = {224, 1, 1};
static unsigned int bfuse_gridDim_40[] = {9340, 1, 1};
static unsigned int bfuse_blockDim_40[] = {64, 1, 1};
static unsigned int bfuse_gridDim_41[] = {44712, 1, 1};
static unsigned int bfuse_blockDim_41[] = {64, 1, 1};
static unsigned int bfuse_gridDim_42[] = {2224, 1, 1};
static unsigned int bfuse_blockDim_42[] = {216, 1, 1};
static unsigned int bfuse_gridDim_43[] = {2496, 1, 1};
static unsigned int bfuse_blockDim_43[] = {104, 1, 1};
static unsigned int bfuse_gridDim_44[] = {3448, 1, 1};
static unsigned int bfuse_blockDim_44[] = {152, 1, 1};
static unsigned int bfuse_gridDim_45[] = {2192, 1, 1};
static unsigned int bfuse_blockDim_45[] = {112, 1, 1};
static unsigned int bfuse_gridDim_46[] = {2192, 1, 1};
static unsigned int bfuse_blockDim_46[] = {112, 1, 1};
static unsigned int bfuse_gridDim_47[] = {2976, 1, 1};
static unsigned int bfuse_blockDim_47[] = {224, 1, 1};
static unsigned int bfuse_gridDim_48[] = {11356, 1, 1};
static unsigned int bfuse_blockDim_48[] = {64, 1, 1};
static unsigned int bfuse_gridDim_49[] = {46728, 1, 1};
static unsigned int bfuse_blockDim_49[] = {64, 1, 1};
static unsigned int bfuse_gridDim_50[] = {4240, 1, 1};
static unsigned int bfuse_blockDim_50[] = {216, 1, 1};
static unsigned int bfuse_gridDim_51[] = {4512, 1, 1};
static unsigned int bfuse_blockDim_51[] = {104, 1, 1};
static unsigned int bfuse_gridDim_52[] = {5464, 1, 1};
static unsigned int bfuse_blockDim_52[] = {152, 1, 1};
static unsigned int bfuse_gridDim_53[] = {4208, 1, 1};
static unsigned int bfuse_blockDim_53[] = {112, 1, 1};
static unsigned int bfuse_gridDim_54[] = {4208, 1, 1};
static unsigned int bfuse_blockDim_54[] = {112, 1, 1};
static unsigned int bfuse_gridDim_55[] = {4992, 1, 1};
static unsigned int bfuse_blockDim_55[] = {224, 1, 1};
static unsigned int bfuse_gridDim_56[] = {7772, 1, 1};
static unsigned int bfuse_blockDim_56[] = {64, 1, 1};
static unsigned int bfuse_gridDim_57[] = {43144, 1, 1};
static unsigned int bfuse_blockDim_57[] = {64, 1, 1};
static unsigned int bfuse_gridDim_58[] = {656, 1, 1};
static unsigned int bfuse_blockDim_58[] = {216, 1, 1};
static unsigned int bfuse_gridDim_59[] = {928, 1, 1};
static unsigned int bfuse_blockDim_59[] = {104, 1, 1};
static unsigned int bfuse_gridDim_60[] = {1880, 1, 1};
static unsigned int bfuse_blockDim_60[] = {152, 1, 1};
static unsigned int bfuse_gridDim_61[] = {624, 1, 1};
static unsigned int bfuse_blockDim_61[] = {112, 1, 1};
static unsigned int bfuse_gridDim_62[] = {624, 1, 1};
static unsigned int bfuse_blockDim_62[] = {112, 1, 1};
static unsigned int bfuse_gridDim_63[] = {1408, 1, 1};
static unsigned int bfuse_blockDim_63[] = {224, 1, 1};
//------------------------------------------------------------------------------------------
extern "C" void conv2d_7(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_5(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_4(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_3(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_2(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_6(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_1(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_0(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4(float* __restrict, float* __restrict, float* __restrict);
//------------------------------------------------------------------------------------------
extern "C" void bgemm_0_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
//------------------------------------------------------------------------------------------
extern "C" void bgemm_2_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
//------------------------------------------------------------------------------------------
#define ASSIGN_KERNEL1(I_shape, F_shape, O_shape, func, gridDim, blockDim, idx) \
  do \
  { \
    switch (idx) \
    { \
    case 0: \
      I_shape  = kernel1_I_shape_0; \
      F_shape  = kernel1_F_shape_0; \
      O_shape  = kernel1_O_shape_0; \
      func     = bgemm_0; \
      gridDim  = kernel1_gridDim_0; \
      blockDim = kernel1_blockDim_0; \
      break; \
    case 1: \
      I_shape  = kernel1_I_shape_1; \
      F_shape  = kernel1_F_shape_1; \
      O_shape  = kernel1_O_shape_1; \
      func     = bgemm_1; \
      gridDim  = kernel1_gridDim_1; \
      blockDim = kernel1_blockDim_1; \
      break; \
    case 2: \
      I_shape  = kernel1_I_shape_2; \
      F_shape  = kernel1_F_shape_2; \
      O_shape  = kernel1_O_shape_2; \
      func     = bgemm_2; \
      gridDim  = kernel1_gridDim_2; \
      blockDim = kernel1_blockDim_2; \
      break; \
    case 3: \
      I_shape  = kernel1_I_shape_3; \
      F_shape  = kernel1_F_shape_3; \
      O_shape  = kernel1_O_shape_3; \
      func     = bgemm_3; \
      gridDim  = kernel1_gridDim_3; \
      blockDim = kernel1_blockDim_3; \
      break; \
    case 4: \
      I_shape  = kernel1_I_shape_4; \
      F_shape  = kernel1_F_shape_4; \
      O_shape  = kernel1_O_shape_4; \
      func     = bgemm_4; \
      gridDim  = kernel1_gridDim_4; \
      blockDim = kernel1_blockDim_4; \
      break; \
    case 5: \
      I_shape  = kernel1_I_shape_5; \
      F_shape  = kernel1_F_shape_5; \
      O_shape  = kernel1_O_shape_5; \
      func     = bgemm_5; \
      gridDim  = kernel1_gridDim_5; \
      blockDim = kernel1_blockDim_5; \
      break; \
    case 6: \
      I_shape  = kernel1_I_shape_6; \
      F_shape  = kernel1_F_shape_6; \
      O_shape  = kernel1_O_shape_6; \
      func     = bgemm_6; \
      gridDim  = kernel1_gridDim_6; \
      blockDim = kernel1_blockDim_6; \
      break; \
    case 7: \
      I_shape  = kernel1_I_shape_7; \
      F_shape  = kernel1_F_shape_7; \
      O_shape  = kernel1_O_shape_7; \
      func     = bgemm_7; \
      gridDim  = kernel1_gridDim_7; \
      blockDim = kernel1_blockDim_7; \
      break; \
    } \
  } while(0)
//------------------------------------------------------------------------------------------
#define ASSIGN_KERNEL2(I_shape, F_shape, O_shape, func, gridDim, blockDim, idx) \
  do \
  { \
    switch (idx) \
    { \
    case 0: \
      I_shape  = kernel2_I_shape_0; \
      F_shape  = kernel2_F_shape_0; \
      O_shape  = kernel2_O_shape_0; \
      func     = conv2d_0; \
      gridDim  = kernel2_gridDim_0; \
      blockDim = kernel2_blockDim_0; \
      break; \
    case 1: \
      I_shape  = kernel2_I_shape_1; \
      F_shape  = kernel2_F_shape_1; \
      O_shape  = kernel2_O_shape_1; \
      func     = conv2d_1; \
      gridDim  = kernel2_gridDim_1; \
      blockDim = kernel2_blockDim_1; \
      break; \
    case 2: \
      I_shape  = kernel2_I_shape_2; \
      F_shape  = kernel2_F_shape_2; \
      O_shape  = kernel2_O_shape_2; \
      func     = conv2d_2; \
      gridDim  = kernel2_gridDim_2; \
      blockDim = kernel2_blockDim_2; \
      break; \
    case 3: \
      I_shape  = kernel2_I_shape_3; \
      F_shape  = kernel2_F_shape_3; \
      O_shape  = kernel2_O_shape_3; \
      func     = conv2d_3; \
      gridDim  = kernel2_gridDim_3; \
      blockDim = kernel2_blockDim_3; \
      break; \
    case 4: \
      I_shape  = kernel2_I_shape_4; \
      F_shape  = kernel2_F_shape_4; \
      O_shape  = kernel2_O_shape_4; \
      func     = conv2d_4; \
      gridDim  = kernel2_gridDim_4; \
      blockDim = kernel2_blockDim_4; \
      break; \
    case 5: \
      I_shape  = kernel2_I_shape_5; \
      F_shape  = kernel2_F_shape_5; \
      O_shape  = kernel2_O_shape_5; \
      func     = conv2d_5; \
      gridDim  = kernel2_gridDim_5; \
      blockDim = kernel2_blockDim_5; \
      break; \
    case 6: \
      I_shape  = kernel2_I_shape_6; \
      F_shape  = kernel2_F_shape_6; \
      O_shape  = kernel2_O_shape_6; \
      func     = conv2d_6; \
      gridDim  = kernel2_gridDim_6; \
      blockDim = kernel2_blockDim_6; \
      break; \
    case 7: \
      I_shape  = kernel2_I_shape_7; \
      F_shape  = kernel2_F_shape_7; \
      O_shape  = kernel2_O_shape_7; \
      func     = conv2d_7; \
      gridDim  = kernel2_gridDim_7; \
      blockDim = kernel2_blockDim_7; \
      break; \
    } \
  } while(0)
//------------------------------------------------------------------------------------------
#define ASSIGN_HFUSE(I1_shape, F1_shape, O1_shape, I2_shape, F2_shpae, O2_shape, func, gridDim, blockDim, idx1, idx2) \
  do \
  { \
    switch (idx1) \
    { \
    case 0: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_0_conv2d_0_fused_hfuse; \
        gridDim  = hfuse_gridDim_0; \
        blockDim = hfuse_blockDim_0; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_0_conv2d_1_fused_hfuse; \
        gridDim  = hfuse_gridDim_1; \
        blockDim = hfuse_blockDim_1; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_0_conv2d_2_fused_hfuse; \
        gridDim  = hfuse_gridDim_2; \
        blockDim = hfuse_blockDim_2; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_0_conv2d_3_fused_hfuse; \
        gridDim  = hfuse_gridDim_3; \
        blockDim = hfuse_blockDim_3; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_0_conv2d_4_fused_hfuse; \
        gridDim  = hfuse_gridDim_4; \
        blockDim = hfuse_blockDim_4; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_0_conv2d_5_fused_hfuse; \
        gridDim  = hfuse_gridDim_5; \
        blockDim = hfuse_blockDim_5; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_0_conv2d_6_fused_hfuse; \
        gridDim  = hfuse_gridDim_6; \
        blockDim = hfuse_blockDim_6; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_0_conv2d_7_fused_hfuse; \
        gridDim  = hfuse_gridDim_7; \
        blockDim = hfuse_blockDim_7; \
        break; \
      } \
      break; \
    case 1: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_1_conv2d_0_fused_hfuse; \
        gridDim  = hfuse_gridDim_8; \
        blockDim = hfuse_blockDim_8; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_1_conv2d_1_fused_hfuse; \
        gridDim  = hfuse_gridDim_9; \
        blockDim = hfuse_blockDim_9; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_1_conv2d_2_fused_hfuse; \
        gridDim  = hfuse_gridDim_10; \
        blockDim = hfuse_blockDim_10; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_1_conv2d_3_fused_hfuse; \
        gridDim  = hfuse_gridDim_11; \
        blockDim = hfuse_blockDim_11; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_1_conv2d_4_fused_hfuse; \
        gridDim  = hfuse_gridDim_12; \
        blockDim = hfuse_blockDim_12; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_1_conv2d_5_fused_hfuse; \
        gridDim  = hfuse_gridDim_13; \
        blockDim = hfuse_blockDim_13; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_1_conv2d_6_fused_hfuse; \
        gridDim  = hfuse_gridDim_14; \
        blockDim = hfuse_blockDim_14; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_1_conv2d_7_fused_hfuse; \
        gridDim  = hfuse_gridDim_15; \
        blockDim = hfuse_blockDim_15; \
        break; \
      } \
      break; \
    case 2: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_2_conv2d_0_fused_hfuse; \
        gridDim  = hfuse_gridDim_16; \
        blockDim = hfuse_blockDim_16; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_2_conv2d_1_fused_hfuse; \
        gridDim  = hfuse_gridDim_17; \
        blockDim = hfuse_blockDim_17; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_2_conv2d_2_fused_hfuse; \
        gridDim  = hfuse_gridDim_18; \
        blockDim = hfuse_blockDim_18; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_2_conv2d_3_fused_hfuse; \
        gridDim  = hfuse_gridDim_19; \
        blockDim = hfuse_blockDim_19; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_2_conv2d_4_fused_hfuse; \
        gridDim  = hfuse_gridDim_20; \
        blockDim = hfuse_blockDim_20; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_2_conv2d_5_fused_hfuse; \
        gridDim  = hfuse_gridDim_21; \
        blockDim = hfuse_blockDim_21; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_2_conv2d_6_fused_hfuse; \
        gridDim  = hfuse_gridDim_22; \
        blockDim = hfuse_blockDim_22; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_2_conv2d_7_fused_hfuse; \
        gridDim  = hfuse_gridDim_23; \
        blockDim = hfuse_blockDim_23; \
        break; \
      } \
      break; \
    case 3: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_3_conv2d_0_fused_hfuse; \
        gridDim  = hfuse_gridDim_24; \
        blockDim = hfuse_blockDim_24; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_3_conv2d_1_fused_hfuse; \
        gridDim  = hfuse_gridDim_25; \
        blockDim = hfuse_blockDim_25; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_3_conv2d_2_fused_hfuse; \
        gridDim  = hfuse_gridDim_26; \
        blockDim = hfuse_blockDim_26; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_3_conv2d_3_fused_hfuse; \
        gridDim  = hfuse_gridDim_27; \
        blockDim = hfuse_blockDim_27; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_3_conv2d_4_fused_hfuse; \
        gridDim  = hfuse_gridDim_28; \
        blockDim = hfuse_blockDim_28; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_3_conv2d_5_fused_hfuse; \
        gridDim  = hfuse_gridDim_29; \
        blockDim = hfuse_blockDim_29; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_3_conv2d_6_fused_hfuse; \
        gridDim  = hfuse_gridDim_30; \
        blockDim = hfuse_blockDim_30; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_3_conv2d_7_fused_hfuse; \
        gridDim  = hfuse_gridDim_31; \
        blockDim = hfuse_blockDim_31; \
        break; \
      } \
      break; \
    case 4: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_4_conv2d_0_fused_hfuse; \
        gridDim  = hfuse_gridDim_32; \
        blockDim = hfuse_blockDim_32; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_4_conv2d_1_fused_hfuse; \
        gridDim  = hfuse_gridDim_33; \
        blockDim = hfuse_blockDim_33; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_4_conv2d_2_fused_hfuse; \
        gridDim  = hfuse_gridDim_34; \
        blockDim = hfuse_blockDim_34; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_4_conv2d_3_fused_hfuse; \
        gridDim  = hfuse_gridDim_35; \
        blockDim = hfuse_blockDim_35; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_4_conv2d_4_fused_hfuse; \
        gridDim  = hfuse_gridDim_36; \
        blockDim = hfuse_blockDim_36; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_4_conv2d_5_fused_hfuse; \
        gridDim  = hfuse_gridDim_37; \
        blockDim = hfuse_blockDim_37; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_4_conv2d_6_fused_hfuse; \
        gridDim  = hfuse_gridDim_38; \
        blockDim = hfuse_blockDim_38; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_4_conv2d_7_fused_hfuse; \
        gridDim  = hfuse_gridDim_39; \
        blockDim = hfuse_blockDim_39; \
        break; \
      } \
      break; \
    case 5: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_5_conv2d_0_fused_hfuse; \
        gridDim  = hfuse_gridDim_40; \
        blockDim = hfuse_blockDim_40; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_5_conv2d_1_fused_hfuse; \
        gridDim  = hfuse_gridDim_41; \
        blockDim = hfuse_blockDim_41; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_5_conv2d_2_fused_hfuse; \
        gridDim  = hfuse_gridDim_42; \
        blockDim = hfuse_blockDim_42; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_5_conv2d_3_fused_hfuse; \
        gridDim  = hfuse_gridDim_43; \
        blockDim = hfuse_blockDim_43; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_5_conv2d_4_fused_hfuse; \
        gridDim  = hfuse_gridDim_44; \
        blockDim = hfuse_blockDim_44; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_5_conv2d_5_fused_hfuse; \
        gridDim  = hfuse_gridDim_45; \
        blockDim = hfuse_blockDim_45; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_5_conv2d_6_fused_hfuse; \
        gridDim  = hfuse_gridDim_46; \
        blockDim = hfuse_blockDim_46; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_5_conv2d_7_fused_hfuse; \
        gridDim  = hfuse_gridDim_47; \
        blockDim = hfuse_blockDim_47; \
        break; \
      } \
      break; \
    case 6: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_6_conv2d_0_fused_hfuse; \
        gridDim  = hfuse_gridDim_48; \
        blockDim = hfuse_blockDim_48; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_6_conv2d_1_fused_hfuse; \
        gridDim  = hfuse_gridDim_49; \
        blockDim = hfuse_blockDim_49; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_6_conv2d_2_fused_hfuse; \
        gridDim  = hfuse_gridDim_50; \
        blockDim = hfuse_blockDim_50; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_6_conv2d_3_fused_hfuse; \
        gridDim  = hfuse_gridDim_51; \
        blockDim = hfuse_blockDim_51; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_6_conv2d_4_fused_hfuse; \
        gridDim  = hfuse_gridDim_52; \
        blockDim = hfuse_blockDim_52; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_6_conv2d_5_fused_hfuse; \
        gridDim  = hfuse_gridDim_53; \
        blockDim = hfuse_blockDim_53; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_6_conv2d_6_fused_hfuse; \
        gridDim  = hfuse_gridDim_54; \
        blockDim = hfuse_blockDim_54; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_6_conv2d_7_fused_hfuse; \
        gridDim  = hfuse_gridDim_55; \
        blockDim = hfuse_blockDim_55; \
        break; \
      } \
      break; \
    case 7: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_7_conv2d_0_fused_hfuse; \
        gridDim  = hfuse_gridDim_56; \
        blockDim = hfuse_blockDim_56; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_7_conv2d_1_fused_hfuse; \
        gridDim  = hfuse_gridDim_57; \
        blockDim = hfuse_blockDim_57; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_7_conv2d_2_fused_hfuse; \
        gridDim  = hfuse_gridDim_58; \
        blockDim = hfuse_blockDim_58; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_7_conv2d_3_fused_hfuse; \
        gridDim  = hfuse_gridDim_59; \
        blockDim = hfuse_blockDim_59; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_7_conv2d_4_fused_hfuse; \
        gridDim  = hfuse_gridDim_60; \
        blockDim = hfuse_blockDim_60; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_7_conv2d_5_fused_hfuse; \
        gridDim  = hfuse_gridDim_61; \
        blockDim = hfuse_blockDim_61; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_7_conv2d_6_fused_hfuse; \
        gridDim  = hfuse_gridDim_62; \
        blockDim = hfuse_blockDim_62; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_7_conv2d_7_fused_hfuse; \
        gridDim  = hfuse_gridDim_63; \
        blockDim = hfuse_blockDim_63; \
        break; \
      } \
      break; \
    } \
  } while(0)
//------------------------------------------------------------------------------------------
#define ASSIGN_BFUSE(I1_shape, F1_shape, O1_shape, I2_shape, F2_shape, O2_shape, func, gridDim, blockDim, idx1, idx2) \
  do \
  { \
    switch (idx1) \
    { \
    case 0: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_0_conv2d_0_fused_bfuse; \
        gridDim  = bfuse_gridDim_0; \
        blockDim = bfuse_blockDim_0; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_0_conv2d_1_fused_bfuse; \
        gridDim  = bfuse_gridDim_1; \
        blockDim = bfuse_blockDim_1; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_0_conv2d_2_fused_bfuse; \
        gridDim  = bfuse_gridDim_2; \
        blockDim = bfuse_blockDim_2; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_0_conv2d_3_fused_bfuse; \
        gridDim  = bfuse_gridDim_3; \
        blockDim = bfuse_blockDim_3; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_0_conv2d_4_fused_bfuse; \
        gridDim  = bfuse_gridDim_4; \
        blockDim = bfuse_blockDim_4; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_0_conv2d_5_fused_bfuse; \
        gridDim  = bfuse_gridDim_5; \
        blockDim = bfuse_blockDim_5; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_0_conv2d_6_fused_bfuse; \
        gridDim  = bfuse_gridDim_6; \
        blockDim = bfuse_blockDim_6; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_0; \
        F1_shape = kernel1_F_shape_0; \
        O1_shape = kernel1_O_shape_0; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_0_conv2d_7_fused_bfuse; \
        gridDim  = bfuse_gridDim_7; \
        blockDim = bfuse_blockDim_7; \
        break; \
      } \
      break; \
    case 1: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_1_conv2d_0_fused_bfuse; \
        gridDim  = bfuse_gridDim_8; \
        blockDim = bfuse_blockDim_8; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_1_conv2d_1_fused_bfuse; \
        gridDim  = bfuse_gridDim_9; \
        blockDim = bfuse_blockDim_9; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_1_conv2d_2_fused_bfuse; \
        gridDim  = bfuse_gridDim_10; \
        blockDim = bfuse_blockDim_10; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_1_conv2d_3_fused_bfuse; \
        gridDim  = bfuse_gridDim_11; \
        blockDim = bfuse_blockDim_11; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_1_conv2d_4_fused_bfuse; \
        gridDim  = bfuse_gridDim_12; \
        blockDim = bfuse_blockDim_12; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_1_conv2d_5_fused_bfuse; \
        gridDim  = bfuse_gridDim_13; \
        blockDim = bfuse_blockDim_13; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_1_conv2d_6_fused_bfuse; \
        gridDim  = bfuse_gridDim_14; \
        blockDim = bfuse_blockDim_14; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_1; \
        F1_shape = kernel1_F_shape_1; \
        O1_shape = kernel1_O_shape_1; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_1_conv2d_7_fused_bfuse; \
        gridDim  = bfuse_gridDim_15; \
        blockDim = bfuse_blockDim_15; \
        break; \
      } \
      break; \
    case 2: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_2_conv2d_0_fused_bfuse; \
        gridDim  = bfuse_gridDim_16; \
        blockDim = bfuse_blockDim_16; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_2_conv2d_1_fused_bfuse; \
        gridDim  = bfuse_gridDim_17; \
        blockDim = bfuse_blockDim_17; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_2_conv2d_2_fused_bfuse; \
        gridDim  = bfuse_gridDim_18; \
        blockDim = bfuse_blockDim_18; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_2_conv2d_3_fused_bfuse; \
        gridDim  = bfuse_gridDim_19; \
        blockDim = bfuse_blockDim_19; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_2_conv2d_4_fused_bfuse; \
        gridDim  = bfuse_gridDim_20; \
        blockDim = bfuse_blockDim_20; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_2_conv2d_5_fused_bfuse; \
        gridDim  = bfuse_gridDim_21; \
        blockDim = bfuse_blockDim_21; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_2_conv2d_6_fused_bfuse; \
        gridDim  = bfuse_gridDim_22; \
        blockDim = bfuse_blockDim_22; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_2; \
        F1_shape = kernel1_F_shape_2; \
        O1_shape = kernel1_O_shape_2; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_2_conv2d_7_fused_bfuse; \
        gridDim  = bfuse_gridDim_23; \
        blockDim = bfuse_blockDim_23; \
        break; \
      } \
      break; \
    case 3: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_3_conv2d_0_fused_bfuse; \
        gridDim  = bfuse_gridDim_24; \
        blockDim = bfuse_blockDim_24; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_3_conv2d_1_fused_bfuse; \
        gridDim  = bfuse_gridDim_25; \
        blockDim = bfuse_blockDim_25; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_3_conv2d_2_fused_bfuse; \
        gridDim  = bfuse_gridDim_26; \
        blockDim = bfuse_blockDim_26; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_3_conv2d_3_fused_bfuse; \
        gridDim  = bfuse_gridDim_27; \
        blockDim = bfuse_blockDim_27; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_3_conv2d_4_fused_bfuse; \
        gridDim  = bfuse_gridDim_28; \
        blockDim = bfuse_blockDim_28; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_3_conv2d_5_fused_bfuse; \
        gridDim  = bfuse_gridDim_29; \
        blockDim = bfuse_blockDim_29; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_3_conv2d_6_fused_bfuse; \
        gridDim  = bfuse_gridDim_30; \
        blockDim = bfuse_blockDim_30; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_3; \
        F1_shape = kernel1_F_shape_3; \
        O1_shape = kernel1_O_shape_3; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_3_conv2d_7_fused_bfuse; \
        gridDim  = bfuse_gridDim_31; \
        blockDim = bfuse_blockDim_31; \
        break; \
      } \
      break; \
    case 4: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_4_conv2d_0_fused_bfuse; \
        gridDim  = bfuse_gridDim_32; \
        blockDim = bfuse_blockDim_32; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_4_conv2d_1_fused_bfuse; \
        gridDim  = bfuse_gridDim_33; \
        blockDim = bfuse_blockDim_33; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_4_conv2d_2_fused_bfuse; \
        gridDim  = bfuse_gridDim_34; \
        blockDim = bfuse_blockDim_34; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_4_conv2d_3_fused_bfuse; \
        gridDim  = bfuse_gridDim_35; \
        blockDim = bfuse_blockDim_35; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_4_conv2d_4_fused_bfuse; \
        gridDim  = bfuse_gridDim_36; \
        blockDim = bfuse_blockDim_36; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_4_conv2d_5_fused_bfuse; \
        gridDim  = bfuse_gridDim_37; \
        blockDim = bfuse_blockDim_37; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_4_conv2d_6_fused_bfuse; \
        gridDim  = bfuse_gridDim_38; \
        blockDim = bfuse_blockDim_38; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_4; \
        F1_shape = kernel1_F_shape_4; \
        O1_shape = kernel1_O_shape_4; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_4_conv2d_7_fused_bfuse; \
        gridDim  = bfuse_gridDim_39; \
        blockDim = bfuse_blockDim_39; \
        break; \
      } \
      break; \
    case 5: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_5_conv2d_0_fused_bfuse; \
        gridDim  = bfuse_gridDim_40; \
        blockDim = bfuse_blockDim_40; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_5_conv2d_1_fused_bfuse; \
        gridDim  = bfuse_gridDim_41; \
        blockDim = bfuse_blockDim_41; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_5_conv2d_2_fused_bfuse; \
        gridDim  = bfuse_gridDim_42; \
        blockDim = bfuse_blockDim_42; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_5_conv2d_3_fused_bfuse; \
        gridDim  = bfuse_gridDim_43; \
        blockDim = bfuse_blockDim_43; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_5_conv2d_4_fused_bfuse; \
        gridDim  = bfuse_gridDim_44; \
        blockDim = bfuse_blockDim_44; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_5_conv2d_5_fused_bfuse; \
        gridDim  = bfuse_gridDim_45; \
        blockDim = bfuse_blockDim_45; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_5_conv2d_6_fused_bfuse; \
        gridDim  = bfuse_gridDim_46; \
        blockDim = bfuse_blockDim_46; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_5; \
        F1_shape = kernel1_F_shape_5; \
        O1_shape = kernel1_O_shape_5; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_5_conv2d_7_fused_bfuse; \
        gridDim  = bfuse_gridDim_47; \
        blockDim = bfuse_blockDim_47; \
        break; \
      } \
      break; \
    case 6: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_6_conv2d_0_fused_bfuse; \
        gridDim  = bfuse_gridDim_48; \
        blockDim = bfuse_blockDim_48; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_6_conv2d_1_fused_bfuse; \
        gridDim  = bfuse_gridDim_49; \
        blockDim = bfuse_blockDim_49; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_6_conv2d_2_fused_bfuse; \
        gridDim  = bfuse_gridDim_50; \
        blockDim = bfuse_blockDim_50; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_6_conv2d_3_fused_bfuse; \
        gridDim  = bfuse_gridDim_51; \
        blockDim = bfuse_blockDim_51; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_6_conv2d_4_fused_bfuse; \
        gridDim  = bfuse_gridDim_52; \
        blockDim = bfuse_blockDim_52; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_6_conv2d_5_fused_bfuse; \
        gridDim  = bfuse_gridDim_53; \
        blockDim = bfuse_blockDim_53; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_6_conv2d_6_fused_bfuse; \
        gridDim  = bfuse_gridDim_54; \
        blockDim = bfuse_blockDim_54; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_6; \
        F1_shape = kernel1_F_shape_6; \
        O1_shape = kernel1_O_shape_6; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_6_conv2d_7_fused_bfuse; \
        gridDim  = bfuse_gridDim_55; \
        blockDim = bfuse_blockDim_55; \
        break; \
      } \
      break; \
    case 7: \
      switch (idx2) \
      { \
      case 0: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_0; \
        F2_shape = kernel2_F_shape_0; \
        O2_shape = kernel2_O_shape_0; \
        func     = bgemm_7_conv2d_0_fused_bfuse; \
        gridDim  = bfuse_gridDim_56; \
        blockDim = bfuse_blockDim_56; \
        break; \
      case 1: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_1; \
        F2_shape = kernel2_F_shape_1; \
        O2_shape = kernel2_O_shape_1; \
        func     = bgemm_7_conv2d_1_fused_bfuse; \
        gridDim  = bfuse_gridDim_57; \
        blockDim = bfuse_blockDim_57; \
        break; \
      case 2: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_2; \
        F2_shape = kernel2_F_shape_2; \
        O2_shape = kernel2_O_shape_2; \
        func     = bgemm_7_conv2d_2_fused_bfuse; \
        gridDim  = bfuse_gridDim_58; \
        blockDim = bfuse_blockDim_58; \
        break; \
      case 3: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_3; \
        F2_shape = kernel2_F_shape_3; \
        O2_shape = kernel2_O_shape_3; \
        func     = bgemm_7_conv2d_3_fused_bfuse; \
        gridDim  = bfuse_gridDim_59; \
        blockDim = bfuse_blockDim_59; \
        break; \
      case 4: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_4; \
        F2_shape = kernel2_F_shape_4; \
        O2_shape = kernel2_O_shape_4; \
        func     = bgemm_7_conv2d_4_fused_bfuse; \
        gridDim  = bfuse_gridDim_60; \
        blockDim = bfuse_blockDim_60; \
        break; \
      case 5: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_5; \
        F2_shape = kernel2_F_shape_5; \
        O2_shape = kernel2_O_shape_5; \
        func     = bgemm_7_conv2d_5_fused_bfuse; \
        gridDim  = bfuse_gridDim_61; \
        blockDim = bfuse_blockDim_61; \
        break; \
      case 6: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_6; \
        F2_shape = kernel2_F_shape_6; \
        O2_shape = kernel2_O_shape_6; \
        func     = bgemm_7_conv2d_6_fused_bfuse; \
        gridDim  = bfuse_gridDim_62; \
        blockDim = bfuse_blockDim_62; \
        break; \
      case 7: \
        I1_shape = kernel1_I_shape_7; \
        F1_shape = kernel1_F_shape_7; \
        O1_shape = kernel1_O_shape_7; \
        I2_shape = kernel2_I_shape_7; \
        F2_shape = kernel2_F_shape_7; \
        O2_shape = kernel2_O_shape_7; \
        func     = bgemm_7_conv2d_7_fused_bfuse; \
        gridDim  = bfuse_gridDim_63; \
        blockDim = bfuse_blockDim_63; \
        break; \
      } \
      break; \
    } \
  } while(0)
//------------------------------------------------------------------------------------------
