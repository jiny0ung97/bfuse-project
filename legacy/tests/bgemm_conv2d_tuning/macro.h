static int kernel1_I_shape_0[] = {256, 512, 64, 1};
static int kernel1_F_shape_0[] = {256, 512, 64, 1};
static int kernel1_O_shape_0[] = {256, 512, 512, 1};
static int kernel1_I_shape_1[] = {256, 512, 512, 1};
static int kernel1_F_shape_1[] = {256, 64, 512, 1};
static int kernel1_O_shape_1[] = {256, 512, 64, 1};
static int kernel1_I_shape_2[] = {256, 256, 80, 1};
static int kernel1_F_shape_2[] = {256, 256, 80, 1};
static int kernel1_O_shape_2[] = {256, 256, 256, 1};
static int kernel1_I_shape_3[] = {256, 256, 256, 1};
static int kernel1_F_shape_3[] = {256, 80, 256, 1};
static int kernel1_O_shape_3[] = {256, 256, 80, 1};
static int kernel1_I_shape_4[] = {256, 208, 80, 1};
static int kernel1_F_shape_4[] = {256, 208, 80, 1};
static int kernel1_O_shape_4[] = {256, 208, 208, 1};
static int kernel1_I_shape_5[] = {256, 208, 208, 1};
static int kernel1_F_shape_5[] = {256, 80, 208, 1};
static int kernel1_O_shape_5[] = {256, 208, 80, 1};
static int kernel1_I_shape_6[] = {256, 1024, 64, 1};
static int kernel1_F_shape_6[] = {256, 512, 64, 1};
static int kernel1_O_shape_6[] = {256, 1024, 512, 1};
static int kernel1_I_shape_7[] = {256, 1024, 512, 1};
static int kernel1_F_shape_7[] = {256, 64, 512, 1};
static int kernel1_O_shape_7[] = {256, 1024, 64, 1};
//------------------------------------------------------------------------------------------
static int kernel2_I_shape_0[] = {256, 64, 112, 112};
static int kernel2_F_shape_0[] = {192, 64, 3, 3};
static int kernel2_O_shape_0[] = {256, 192, 55, 55};
static int kernel2_I_shape_1[] = {256, 32, 147, 147};
static int kernel2_F_shape_1[] = {64, 32, 3, 3};
static int kernel2_O_shape_1[] = {256, 64, 73, 73};
static int kernel2_I_shape_2[] = {256, 64, 56, 56};
static int kernel2_F_shape_2[] = {128, 64, 3, 3};
static int kernel2_O_shape_2[] = {256, 128, 54, 54};
static int kernel2_I_shape_3[] = {256, 128, 28, 28};
static int kernel2_F_shape_3[] = {256, 128, 3, 3};
static int kernel2_O_shape_3[] = {256, 256, 26, 26};
static int kernel2_I_shape_4[] = {256, 16, 227, 227};
static int kernel2_F_shape_4[] = {64, 16, 3, 3};
static int kernel2_O_shape_4[] = {256, 64, 57, 57};
static int kernel2_I_shape_5[] = {256, 64, 56, 56};
static int kernel2_F_shape_5[] = {64, 64, 1, 1};
static int kernel2_O_shape_5[] = {256, 64, 56, 56};
static int kernel2_I_shape_6[] = {256, 64, 56, 56};
static int kernel2_F_shape_6[] = {64, 64, 1, 1};
static int kernel2_O_shape_6[] = {256, 64, 56, 56};
static int kernel2_I_shape_7[] = {256, 256, 56, 56};
static int kernel2_F_shape_7[] = {256, 256, 1, 1};
static int kernel2_O_shape_7[] = {256, 256, 56, 56};
//------------------------------------------------------------------------------------------
static unsigned int kernel1_gridDim_0[] = {8192, 1, 1};
static unsigned int kernel1_blockDim_0[] = {128, 1, 1};
static unsigned int kernel1_gridDim_1[] = {512, 1, 1};
static unsigned int kernel1_blockDim_1[] = {256, 1, 1};
static unsigned int kernel1_gridDim_2[] = {2048, 1, 1};
static unsigned int kernel1_blockDim_2[] = {256, 1, 1};
static unsigned int kernel1_gridDim_3[] = {2048, 1, 1};
static unsigned int kernel1_blockDim_3[] = {80, 1, 1};
static unsigned int kernel1_gridDim_4[] = {2048, 1, 1};
static unsigned int kernel1_blockDim_4[] = {104, 1, 1};
static unsigned int kernel1_gridDim_5[] = {1024, 1, 1};
static unsigned int kernel1_blockDim_5[] = {104, 1, 1};
static unsigned int kernel1_gridDim_6[] = {32768, 1, 1};
static unsigned int kernel1_blockDim_6[] = {128, 1, 1};
static unsigned int kernel1_gridDim_7[] = {4096, 1, 1};
static unsigned int kernel1_blockDim_7[] = {64, 1, 1};
//------------------------------------------------------------------------------------------
static unsigned int kernel2_gridDim_0[] = {7744, 1, 1};
static unsigned int kernel2_blockDim_0[] = {300, 1, 1};
static unsigned int kernel2_gridDim_1[] = {4672, 1, 1};
static unsigned int kernel2_blockDim_1[] = {292, 1, 1};
static unsigned int kernel2_gridDim_2[] = {5184, 1, 1};
static unsigned int kernel2_blockDim_2[] = {256, 1, 1};
static unsigned int kernel2_gridDim_3[] = {1664, 1, 1};
static unsigned int kernel2_blockDim_3[] = {256, 1, 1};
static unsigned int kernel2_gridDim_4[] = {2432, 1, 1};
static unsigned int kernel2_blockDim_4[] = {228, 1, 1};
static unsigned int kernel2_gridDim_5[] = {3584, 1, 1};
static unsigned int kernel2_blockDim_5[] = {128, 1, 1};
static unsigned int kernel2_gridDim_6[] = {12544, 1, 1};
static unsigned int kernel2_blockDim_6[] = {64, 1, 1};
static unsigned int kernel2_gridDim_7[] = {7168, 1, 1};
static unsigned int kernel2_blockDim_7[] = {224, 1, 1};
//------------------------------------------------------------------------------------------
static unsigned int hfuse_gridDim_0[] = {8192, 1, 1};
static unsigned int hfuse_blockDim_0[] = {448, 1, 1};
static unsigned int hfuse_gridDim_1[] = {8192, 1, 1};
static unsigned int hfuse_blockDim_1[] = {448, 1, 1};
static unsigned int hfuse_gridDim_2[] = {8192, 1, 1};
static unsigned int hfuse_blockDim_2[] = {384, 1, 1};
static unsigned int hfuse_gridDim_3[] = {8192, 1, 1};
static unsigned int hfuse_blockDim_3[] = {384, 1, 1};
static unsigned int hfuse_gridDim_4[] = {8192, 1, 1};
static unsigned int hfuse_blockDim_4[] = {384, 1, 1};
static unsigned int hfuse_gridDim_5[] = {8192, 1, 1};
static unsigned int hfuse_blockDim_5[] = {256, 1, 1};
static unsigned int hfuse_gridDim_6[] = {12544, 1, 1};
static unsigned int hfuse_blockDim_6[] = {192, 1, 1};
static unsigned int hfuse_gridDim_7[] = {8192, 1, 1};
static unsigned int hfuse_blockDim_7[] = {352, 1, 1};
static unsigned int hfuse_gridDim_8[] = {7744, 1, 1};
static unsigned int hfuse_blockDim_8[] = {576, 1, 1};
static unsigned int hfuse_gridDim_9[] = {4672, 1, 1};
static unsigned int hfuse_blockDim_9[] = {576, 1, 1};
static unsigned int hfuse_gridDim_10[] = {5184, 1, 1};
static unsigned int hfuse_blockDim_10[] = {512, 1, 1};
static unsigned int hfuse_gridDim_11[] = {1664, 1, 1};
static unsigned int hfuse_blockDim_11[] = {512, 1, 1};
static unsigned int hfuse_gridDim_12[] = {2432, 1, 1};
static unsigned int hfuse_blockDim_12[] = {512, 1, 1};
static unsigned int hfuse_gridDim_13[] = {3584, 1, 1};
static unsigned int hfuse_blockDim_13[] = {384, 1, 1};
static unsigned int hfuse_gridDim_14[] = {12544, 1, 1};
static unsigned int hfuse_blockDim_14[] = {320, 1, 1};
static unsigned int hfuse_gridDim_15[] = {7168, 1, 1};
static unsigned int hfuse_blockDim_15[] = {480, 1, 1};
static unsigned int hfuse_gridDim_16[] = {7744, 1, 1};
static unsigned int hfuse_blockDim_16[] = {576, 1, 1};
static unsigned int hfuse_gridDim_17[] = {4672, 1, 1};
static unsigned int hfuse_blockDim_17[] = {576, 1, 1};
static unsigned int hfuse_gridDim_18[] = {5184, 1, 1};
static unsigned int hfuse_blockDim_18[] = {512, 1, 1};
static unsigned int hfuse_gridDim_19[] = {2048, 1, 1};
static unsigned int hfuse_blockDim_19[] = {512, 1, 1};
static unsigned int hfuse_gridDim_20[] = {2432, 1, 1};
static unsigned int hfuse_blockDim_20[] = {512, 1, 1};
static unsigned int hfuse_gridDim_21[] = {3584, 1, 1};
static unsigned int hfuse_blockDim_21[] = {384, 1, 1};
static unsigned int hfuse_gridDim_22[] = {12544, 1, 1};
static unsigned int hfuse_blockDim_22[] = {320, 1, 1};
static unsigned int hfuse_gridDim_23[] = {7168, 1, 1};
static unsigned int hfuse_blockDim_23[] = {480, 1, 1};
static unsigned int hfuse_gridDim_24[] = {7744, 1, 1};
static unsigned int hfuse_blockDim_24[] = {416, 1, 1};
static unsigned int hfuse_gridDim_25[] = {4672, 1, 1};
static unsigned int hfuse_blockDim_25[] = {416, 1, 1};
static unsigned int hfuse_gridDim_26[] = {5184, 1, 1};
static unsigned int hfuse_blockDim_26[] = {352, 1, 1};
static unsigned int hfuse_gridDim_27[] = {2048, 1, 1};
static unsigned int hfuse_blockDim_27[] = {352, 1, 1};
static unsigned int hfuse_gridDim_28[] = {2432, 1, 1};
static unsigned int hfuse_blockDim_28[] = {352, 1, 1};
static unsigned int hfuse_gridDim_29[] = {3584, 1, 1};
static unsigned int hfuse_blockDim_29[] = {224, 1, 1};
static unsigned int hfuse_gridDim_30[] = {12544, 1, 1};
static unsigned int hfuse_blockDim_30[] = {160, 1, 1};
static unsigned int hfuse_gridDim_31[] = {7168, 1, 1};
static unsigned int hfuse_blockDim_31[] = {320, 1, 1};
static unsigned int hfuse_gridDim_32[] = {7744, 1, 1};
static unsigned int hfuse_blockDim_32[] = {448, 1, 1};
static unsigned int hfuse_gridDim_33[] = {4672, 1, 1};
static unsigned int hfuse_blockDim_33[] = {448, 1, 1};
static unsigned int hfuse_gridDim_34[] = {5184, 1, 1};
static unsigned int hfuse_blockDim_34[] = {384, 1, 1};
static unsigned int hfuse_gridDim_35[] = {2048, 1, 1};
static unsigned int hfuse_blockDim_35[] = {384, 1, 1};
static unsigned int hfuse_gridDim_36[] = {2432, 1, 1};
static unsigned int hfuse_blockDim_36[] = {384, 1, 1};
static unsigned int hfuse_gridDim_37[] = {3584, 1, 1};
static unsigned int hfuse_blockDim_37[] = {256, 1, 1};
static unsigned int hfuse_gridDim_38[] = {12544, 1, 1};
static unsigned int hfuse_blockDim_38[] = {192, 1, 1};
static unsigned int hfuse_gridDim_39[] = {7168, 1, 1};
static unsigned int hfuse_blockDim_39[] = {352, 1, 1};
static unsigned int hfuse_gridDim_40[] = {7744, 1, 1};
static unsigned int hfuse_blockDim_40[] = {448, 1, 1};
static unsigned int hfuse_gridDim_41[] = {4672, 1, 1};
static unsigned int hfuse_blockDim_41[] = {448, 1, 1};
static unsigned int hfuse_gridDim_42[] = {5184, 1, 1};
static unsigned int hfuse_blockDim_42[] = {384, 1, 1};
static unsigned int hfuse_gridDim_43[] = {1664, 1, 1};
static unsigned int hfuse_blockDim_43[] = {384, 1, 1};
static unsigned int hfuse_gridDim_44[] = {2432, 1, 1};
static unsigned int hfuse_blockDim_44[] = {384, 1, 1};
static unsigned int hfuse_gridDim_45[] = {3584, 1, 1};
static unsigned int hfuse_blockDim_45[] = {256, 1, 1};
static unsigned int hfuse_gridDim_46[] = {12544, 1, 1};
static unsigned int hfuse_blockDim_46[] = {192, 1, 1};
static unsigned int hfuse_gridDim_47[] = {7168, 1, 1};
static unsigned int hfuse_blockDim_47[] = {352, 1, 1};
static unsigned int hfuse_gridDim_48[] = {32768, 1, 1};
static unsigned int hfuse_blockDim_48[] = {448, 1, 1};
static unsigned int hfuse_gridDim_49[] = {32768, 1, 1};
static unsigned int hfuse_blockDim_49[] = {448, 1, 1};
static unsigned int hfuse_gridDim_50[] = {32768, 1, 1};
static unsigned int hfuse_blockDim_50[] = {384, 1, 1};
static unsigned int hfuse_gridDim_51[] = {32768, 1, 1};
static unsigned int hfuse_blockDim_51[] = {384, 1, 1};
static unsigned int hfuse_gridDim_52[] = {32768, 1, 1};
static unsigned int hfuse_blockDim_52[] = {384, 1, 1};
static unsigned int hfuse_gridDim_53[] = {32768, 1, 1};
static unsigned int hfuse_blockDim_53[] = {256, 1, 1};
static unsigned int hfuse_gridDim_54[] = {32768, 1, 1};
static unsigned int hfuse_blockDim_54[] = {192, 1, 1};
static unsigned int hfuse_gridDim_55[] = {32768, 1, 1};
static unsigned int hfuse_blockDim_55[] = {352, 1, 1};
static unsigned int hfuse_gridDim_56[] = {7744, 1, 1};
static unsigned int hfuse_blockDim_56[] = {384, 1, 1};
static unsigned int hfuse_gridDim_57[] = {4672, 1, 1};
static unsigned int hfuse_blockDim_57[] = {384, 1, 1};
static unsigned int hfuse_gridDim_58[] = {5184, 1, 1};
static unsigned int hfuse_blockDim_58[] = {320, 1, 1};
static unsigned int hfuse_gridDim_59[] = {4096, 1, 1};
static unsigned int hfuse_blockDim_59[] = {320, 1, 1};
static unsigned int hfuse_gridDim_60[] = {4096, 1, 1};
static unsigned int hfuse_blockDim_60[] = {320, 1, 1};
static unsigned int hfuse_gridDim_61[] = {4096, 1, 1};
static unsigned int hfuse_blockDim_61[] = {192, 1, 1};
static unsigned int hfuse_gridDim_62[] = {12544, 1, 1};
static unsigned int hfuse_blockDim_62[] = {128, 1, 1};
static unsigned int hfuse_gridDim_63[] = {7168, 1, 1};
static unsigned int hfuse_blockDim_63[] = {288, 1, 1};
//------------------------------------------------------------------------------------------
static unsigned int bfuse_gridDim_0[] = {15936, 1, 1};
static unsigned int bfuse_blockDim_0[] = {300, 1, 1};
static unsigned int bfuse_gridDim_1[] = {12864, 1, 1};
static unsigned int bfuse_blockDim_1[] = {292, 1, 1};
static unsigned int bfuse_gridDim_2[] = {13376, 1, 1};
static unsigned int bfuse_blockDim_2[] = {256, 1, 1};
static unsigned int bfuse_gridDim_3[] = {9856, 1, 1};
static unsigned int bfuse_blockDim_3[] = {256, 1, 1};
static unsigned int bfuse_gridDim_4[] = {10624, 1, 1};
static unsigned int bfuse_blockDim_4[] = {228, 1, 1};
static unsigned int bfuse_gridDim_5[] = {11776, 1, 1};
static unsigned int bfuse_blockDim_5[] = {128, 1, 1};
static unsigned int bfuse_gridDim_6[] = {20736, 1, 1};
static unsigned int bfuse_blockDim_6[] = {128, 1, 1};
static unsigned int bfuse_gridDim_7[] = {15360, 1, 1};
static unsigned int bfuse_blockDim_7[] = {224, 1, 1};
static unsigned int bfuse_gridDim_8[] = {8256, 1, 1};
static unsigned int bfuse_blockDim_8[] = {300, 1, 1};
static unsigned int bfuse_gridDim_9[] = {5184, 1, 1};
static unsigned int bfuse_blockDim_9[] = {292, 1, 1};
static unsigned int bfuse_gridDim_10[] = {5696, 1, 1};
static unsigned int bfuse_blockDim_10[] = {256, 1, 1};
static unsigned int bfuse_gridDim_11[] = {2176, 1, 1};
static unsigned int bfuse_blockDim_11[] = {256, 1, 1};
static unsigned int bfuse_gridDim_12[] = {2944, 1, 1};
static unsigned int bfuse_blockDim_12[] = {256, 1, 1};
static unsigned int bfuse_gridDim_13[] = {4096, 1, 1};
static unsigned int bfuse_blockDim_13[] = {256, 1, 1};
static unsigned int bfuse_gridDim_14[] = {13056, 1, 1};
static unsigned int bfuse_blockDim_14[] = {256, 1, 1};
static unsigned int bfuse_gridDim_15[] = {7680, 1, 1};
static unsigned int bfuse_blockDim_15[] = {256, 1, 1};
static unsigned int bfuse_gridDim_16[] = {9792, 1, 1};
static unsigned int bfuse_blockDim_16[] = {300, 1, 1};
static unsigned int bfuse_gridDim_17[] = {6720, 1, 1};
static unsigned int bfuse_blockDim_17[] = {292, 1, 1};
static unsigned int bfuse_gridDim_18[] = {7232, 1, 1};
static unsigned int bfuse_blockDim_18[] = {256, 1, 1};
static unsigned int bfuse_gridDim_19[] = {3712, 1, 1};
static unsigned int bfuse_blockDim_19[] = {256, 1, 1};
static unsigned int bfuse_gridDim_20[] = {4480, 1, 1};
static unsigned int bfuse_blockDim_20[] = {256, 1, 1};
static unsigned int bfuse_gridDim_21[] = {5632, 1, 1};
static unsigned int bfuse_blockDim_21[] = {256, 1, 1};
static unsigned int bfuse_gridDim_22[] = {14592, 1, 1};
static unsigned int bfuse_blockDim_22[] = {256, 1, 1};
static unsigned int bfuse_gridDim_23[] = {9216, 1, 1};
static unsigned int bfuse_blockDim_23[] = {256, 1, 1};
static unsigned int bfuse_gridDim_24[] = {9792, 1, 1};
static unsigned int bfuse_blockDim_24[] = {300, 1, 1};
static unsigned int bfuse_gridDim_25[] = {6720, 1, 1};
static unsigned int bfuse_blockDim_25[] = {292, 1, 1};
static unsigned int bfuse_gridDim_26[] = {7232, 1, 1};
static unsigned int bfuse_blockDim_26[] = {256, 1, 1};
static unsigned int bfuse_gridDim_27[] = {3712, 1, 1};
static unsigned int bfuse_blockDim_27[] = {256, 1, 1};
static unsigned int bfuse_gridDim_28[] = {4480, 1, 1};
static unsigned int bfuse_blockDim_28[] = {228, 1, 1};
static unsigned int bfuse_gridDim_29[] = {5632, 1, 1};
static unsigned int bfuse_blockDim_29[] = {128, 1, 1};
static unsigned int bfuse_gridDim_30[] = {14592, 1, 1};
static unsigned int bfuse_blockDim_30[] = {80, 1, 1};
static unsigned int bfuse_gridDim_31[] = {9216, 1, 1};
static unsigned int bfuse_blockDim_31[] = {224, 1, 1};
static unsigned int bfuse_gridDim_32[] = {9792, 1, 1};
static unsigned int bfuse_blockDim_32[] = {300, 1, 1};
static unsigned int bfuse_gridDim_33[] = {6720, 1, 1};
static unsigned int bfuse_blockDim_33[] = {292, 1, 1};
static unsigned int bfuse_gridDim_34[] = {7232, 1, 1};
static unsigned int bfuse_blockDim_34[] = {256, 1, 1};
static unsigned int bfuse_gridDim_35[] = {3712, 1, 1};
static unsigned int bfuse_blockDim_35[] = {256, 1, 1};
static unsigned int bfuse_gridDim_36[] = {4480, 1, 1};
static unsigned int bfuse_blockDim_36[] = {228, 1, 1};
static unsigned int bfuse_gridDim_37[] = {5632, 1, 1};
static unsigned int bfuse_blockDim_37[] = {128, 1, 1};
static unsigned int bfuse_gridDim_38[] = {14592, 1, 1};
static unsigned int bfuse_blockDim_38[] = {104, 1, 1};
static unsigned int bfuse_gridDim_39[] = {9216, 1, 1};
static unsigned int bfuse_blockDim_39[] = {224, 1, 1};
static unsigned int bfuse_gridDim_40[] = {8768, 1, 1};
static unsigned int bfuse_blockDim_40[] = {300, 1, 1};
static unsigned int bfuse_gridDim_41[] = {5696, 1, 1};
static unsigned int bfuse_blockDim_41[] = {292, 1, 1};
static unsigned int bfuse_gridDim_42[] = {6208, 1, 1};
static unsigned int bfuse_blockDim_42[] = {256, 1, 1};
static unsigned int bfuse_gridDim_43[] = {2688, 1, 1};
static unsigned int bfuse_blockDim_43[] = {256, 1, 1};
static unsigned int bfuse_gridDim_44[] = {3456, 1, 1};
static unsigned int bfuse_blockDim_44[] = {228, 1, 1};
static unsigned int bfuse_gridDim_45[] = {4608, 1, 1};
static unsigned int bfuse_blockDim_45[] = {128, 1, 1};
static unsigned int bfuse_gridDim_46[] = {13568, 1, 1};
static unsigned int bfuse_blockDim_46[] = {104, 1, 1};
static unsigned int bfuse_gridDim_47[] = {8192, 1, 1};
static unsigned int bfuse_blockDim_47[] = {224, 1, 1};
static unsigned int bfuse_gridDim_48[] = {40512, 1, 1};
static unsigned int bfuse_blockDim_48[] = {300, 1, 1};
static unsigned int bfuse_gridDim_49[] = {37440, 1, 1};
static unsigned int bfuse_blockDim_49[] = {292, 1, 1};
static unsigned int bfuse_gridDim_50[] = {37952, 1, 1};
static unsigned int bfuse_blockDim_50[] = {256, 1, 1};
static unsigned int bfuse_gridDim_51[] = {34432, 1, 1};
static unsigned int bfuse_blockDim_51[] = {256, 1, 1};
static unsigned int bfuse_gridDim_52[] = {35200, 1, 1};
static unsigned int bfuse_blockDim_52[] = {228, 1, 1};
static unsigned int bfuse_gridDim_53[] = {36352, 1, 1};
static unsigned int bfuse_blockDim_53[] = {128, 1, 1};
static unsigned int bfuse_gridDim_54[] = {45312, 1, 1};
static unsigned int bfuse_blockDim_54[] = {128, 1, 1};
static unsigned int bfuse_gridDim_55[] = {39936, 1, 1};
static unsigned int bfuse_blockDim_55[] = {224, 1, 1};
static unsigned int bfuse_gridDim_56[] = {11840, 1, 1};
static unsigned int bfuse_blockDim_56[] = {300, 1, 1};
static unsigned int bfuse_gridDim_57[] = {8768, 1, 1};
static unsigned int bfuse_blockDim_57[] = {292, 1, 1};
static unsigned int bfuse_gridDim_58[] = {9280, 1, 1};
static unsigned int bfuse_blockDim_58[] = {256, 1, 1};
static unsigned int bfuse_gridDim_59[] = {5760, 1, 1};
static unsigned int bfuse_blockDim_59[] = {256, 1, 1};
static unsigned int bfuse_gridDim_60[] = {6528, 1, 1};
static unsigned int bfuse_blockDim_60[] = {228, 1, 1};
static unsigned int bfuse_gridDim_61[] = {7680, 1, 1};
static unsigned int bfuse_blockDim_61[] = {128, 1, 1};
static unsigned int bfuse_gridDim_62[] = {16640, 1, 1};
static unsigned int bfuse_blockDim_62[] = {64, 1, 1};
static unsigned int bfuse_gridDim_63[] = {11264, 1, 1};
static unsigned int bfuse_blockDim_63[] = {224, 1, 1};
//------------------------------------------------------------------------------------------
extern "C" void bgemm_7(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_1(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_2(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_0(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_6(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_5(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_3(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_7(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_4(float* __restrict, float* __restrict, float* __restrict);
//------------------------------------------------------------------------------------------
extern "C" void bgemm_3_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_5_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_5_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_3_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_1_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_0_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_1_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_7_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_2_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_2_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_4_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_0_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_5_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_0_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_1_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_5_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_2_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_2_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_0_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_1_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_0_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_1_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_1_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_1_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_1_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_6_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_4_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
// extern "C" void bgemm_7_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
//------------------------------------------------------------------------------------------
extern "C" void bgemm_1_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
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
