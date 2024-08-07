static int kernel1_I_shape_0[] = {1024, 512, 64, 1};
static int kernel1_F_shape_0[] = {1024, 512, 64, 1};
static int kernel1_O_shape_0[] = {1024, 512, 512, 1};
static int kernel1_I_shape_1[] = {1024, 512, 512, 1};
static int kernel1_F_shape_1[] = {1024, 64, 512, 1};
static int kernel1_O_shape_1[] = {1024, 512, 64, 1};
static int kernel1_I_shape_2[] = {1024, 256, 80, 1};
static int kernel1_F_shape_2[] = {1024, 256, 80, 1};
static int kernel1_O_shape_2[] = {1024, 256, 256, 1};
static int kernel1_I_shape_3[] = {1024, 256, 256, 1};
static int kernel1_F_shape_3[] = {1024, 80, 256, 1};
static int kernel1_O_shape_3[] = {1024, 256, 80, 1};
static int kernel1_I_shape_4[] = {1024, 208, 80, 1};
static int kernel1_F_shape_4[] = {1024, 208, 80, 1};
static int kernel1_O_shape_4[] = {1024, 208, 208, 1};
static int kernel1_I_shape_5[] = {1024, 208, 208, 1};
static int kernel1_F_shape_5[] = {1024, 80, 208, 1};
static int kernel1_O_shape_5[] = {1024, 208, 80, 1};
static int kernel1_I_shape_6[] = {1024, 1024, 64, 1};
static int kernel1_F_shape_6[] = {1024, 512, 64, 1};
static int kernel1_O_shape_6[] = {1024, 1024, 512, 1};
static int kernel1_I_shape_7[] = {1024, 1024, 512, 1};
static int kernel1_F_shape_7[] = {1024, 64, 512, 1};
static int kernel1_O_shape_7[] = {1024, 1024, 64, 1};
//------------------------------------------------------------------------------------------
static int kernel2_I_shape_0[] = {1024, 64, 112, 112};
static int kernel2_F_shape_0[] = {192, 64, 3, 3};
static int kernel2_O_shape_0[] = {1024, 192, 55, 55};
static int kernel2_I_shape_1[] = {1024, 32, 147, 147};
static int kernel2_F_shape_1[] = {64, 32, 3, 3};
static int kernel2_O_shape_1[] = {1024, 64, 73, 73};
static int kernel2_I_shape_2[] = {1024, 64, 56, 56};
static int kernel2_F_shape_2[] = {128, 64, 3, 3};
static int kernel2_O_shape_2[] = {1024, 128, 54, 54};
static int kernel2_I_shape_3[] = {1024, 128, 28, 28};
static int kernel2_F_shape_3[] = {256, 128, 3, 3};
static int kernel2_O_shape_3[] = {1024, 256, 26, 26};
static int kernel2_I_shape_4[] = {1024, 16, 227, 227};
static int kernel2_F_shape_4[] = {64, 16, 3, 3};
static int kernel2_O_shape_4[] = {1024, 64, 57, 57};
static int kernel2_I_shape_5[] = {1024, 64, 56, 56};
static int kernel2_F_shape_5[] = {64, 64, 1, 1};
static int kernel2_O_shape_5[] = {1024, 64, 56, 56};
static int kernel2_I_shape_6[] = {1024, 64, 56, 56};
static int kernel2_F_shape_6[] = {64, 64, 1, 1};
static int kernel2_O_shape_6[] = {1024, 64, 56, 56};
static int kernel2_I_shape_7[] = {1024, 256, 56, 56};
static int kernel2_F_shape_7[] = {256, 256, 1, 1};
static int kernel2_O_shape_7[] = {1024, 256, 56, 56};
//------------------------------------------------------------------------------------------
static unsigned int kernel1_gridDim_0[] = {8, 8, 1024};
static unsigned int kernel1_blockDim_0[] = {8, 8, 1};
static unsigned int kernel1_gridDim_1[] = {1, 8, 1024};
static unsigned int kernel1_blockDim_1[] = {8, 8, 1};
static unsigned int kernel1_gridDim_2[] = {4, 4, 1024};
static unsigned int kernel1_blockDim_2[] = {8, 8, 1};
static unsigned int kernel1_gridDim_3[] = {5, 4, 1024};
static unsigned int kernel1_blockDim_3[] = {8, 8, 1};
static unsigned int kernel1_gridDim_4[] = {13, 13, 1024};
static unsigned int kernel1_blockDim_4[] = {8, 8, 1};
static unsigned int kernel1_gridDim_5[] = {5, 13, 1024};
static unsigned int kernel1_blockDim_5[] = {8, 8, 1};
static unsigned int kernel1_gridDim_6[] = {8, 16, 1024};
static unsigned int kernel1_blockDim_6[] = {8, 8, 1};
static unsigned int kernel1_gridDim_7[] = {1, 16, 1024};
static unsigned int kernel1_blockDim_7[] = {8, 8, 1};
//------------------------------------------------------------------------------------------
static unsigned int kernel2_gridDim_0[] = {55, 11, 3072};
static unsigned int kernel2_blockDim_0[] = {1, 1, 16};
static unsigned int kernel2_gridDim_1[] = {73, 73, 2048};
static unsigned int kernel2_blockDim_1[] = {1, 1, 8};
static unsigned int kernel2_gridDim_2[] = {2, 9, 2048};
static unsigned int kernel2_blockDim_2[] = {27, 1, 8};
static unsigned int kernel2_gridDim_3[] = {2, 13, 4096};
static unsigned int kernel2_blockDim_3[] = {13, 1, 8};
static unsigned int kernel2_gridDim_4[] = {3, 57, 2048};
static unsigned int kernel2_blockDim_4[] = {19, 1, 8};
static unsigned int kernel2_gridDim_5[] = {1, 28, 1024};
static unsigned int kernel2_blockDim_5[] = {56, 1, 2};
static unsigned int kernel2_gridDim_6[] = {1, 28, 1024};
static unsigned int kernel2_blockDim_6[] = {56, 1, 2};
static unsigned int kernel2_gridDim_7[] = {2, 28, 4096};
static unsigned int kernel2_blockDim_7[] = {14, 1, 16};
//------------------------------------------------------------------------------------------
static unsigned int hfuse_gridDim_0[] = {1858560, 1, 1};
static unsigned int hfuse_blockDim_0[] = {96, 1, 1};
static unsigned int hfuse_gridDim_1[] = {10913792, 1, 1};
static unsigned int hfuse_blockDim_1[] = {96, 1, 1};
static unsigned int hfuse_gridDim_2[] = {65536, 1, 1};
static unsigned int hfuse_blockDim_2[] = {288, 1, 1};
static unsigned int hfuse_gridDim_3[] = {106496, 1, 1};
static unsigned int hfuse_blockDim_3[] = {192, 1, 1};
static unsigned int hfuse_gridDim_4[] = {350208, 1, 1};
static unsigned int hfuse_blockDim_4[] = {224, 1, 1};
static unsigned int hfuse_gridDim_5[] = {65536, 1, 1};
static unsigned int hfuse_blockDim_5[] = {192, 1, 1};
static unsigned int hfuse_gridDim_6[] = {65536, 1, 1};
static unsigned int hfuse_blockDim_6[] = {192, 1, 1};
static unsigned int hfuse_gridDim_7[] = {229376, 1, 1};
static unsigned int hfuse_blockDim_7[] = {288, 1, 1};
static unsigned int hfuse_gridDim_8[] = {1858560, 1, 1};
static unsigned int hfuse_blockDim_8[] = {96, 1, 1};
static unsigned int hfuse_gridDim_9[] = {10913792, 1, 1};
static unsigned int hfuse_blockDim_9[] = {96, 1, 1};
static unsigned int hfuse_gridDim_10[] = {36864, 1, 1};
static unsigned int hfuse_blockDim_10[] = {288, 1, 1};
static unsigned int hfuse_gridDim_11[] = {106496, 1, 1};
static unsigned int hfuse_blockDim_11[] = {192, 1, 1};
static unsigned int hfuse_gridDim_12[] = {350208, 1, 1};
static unsigned int hfuse_blockDim_12[] = {224, 1, 1};
static unsigned int hfuse_gridDim_13[] = {28672, 1, 1};
static unsigned int hfuse_blockDim_13[] = {192, 1, 1};
static unsigned int hfuse_gridDim_14[] = {28672, 1, 1};
static unsigned int hfuse_blockDim_14[] = {192, 1, 1};
static unsigned int hfuse_gridDim_15[] = {229376, 1, 1};
static unsigned int hfuse_blockDim_15[] = {288, 1, 1};
static unsigned int hfuse_gridDim_16[] = {1858560, 1, 1};
static unsigned int hfuse_blockDim_16[] = {96, 1, 1};
static unsigned int hfuse_gridDim_17[] = {10913792, 1, 1};
static unsigned int hfuse_blockDim_17[] = {96, 1, 1};
static unsigned int hfuse_gridDim_18[] = {36864, 1, 1};
static unsigned int hfuse_blockDim_18[] = {288, 1, 1};
static unsigned int hfuse_gridDim_19[] = {106496, 1, 1};
static unsigned int hfuse_blockDim_19[] = {192, 1, 1};
static unsigned int hfuse_gridDim_20[] = {350208, 1, 1};
static unsigned int hfuse_blockDim_20[] = {224, 1, 1};
static unsigned int hfuse_gridDim_21[] = {28672, 1, 1};
static unsigned int hfuse_blockDim_21[] = {192, 1, 1};
static unsigned int hfuse_gridDim_22[] = {28672, 1, 1};
static unsigned int hfuse_blockDim_22[] = {192, 1, 1};
static unsigned int hfuse_gridDim_23[] = {229376, 1, 1};
static unsigned int hfuse_blockDim_23[] = {288, 1, 1};
static unsigned int hfuse_gridDim_24[] = {1858560, 1, 1};
static unsigned int hfuse_blockDim_24[] = {96, 1, 1};
static unsigned int hfuse_gridDim_25[] = {10913792, 1, 1};
static unsigned int hfuse_blockDim_25[] = {96, 1, 1};
static unsigned int hfuse_gridDim_26[] = {36864, 1, 1};
static unsigned int hfuse_blockDim_26[] = {288, 1, 1};
static unsigned int hfuse_gridDim_27[] = {106496, 1, 1};
static unsigned int hfuse_blockDim_27[] = {192, 1, 1};
static unsigned int hfuse_gridDim_28[] = {350208, 1, 1};
static unsigned int hfuse_blockDim_28[] = {224, 1, 1};
static unsigned int hfuse_gridDim_29[] = {28672, 1, 1};
static unsigned int hfuse_blockDim_29[] = {192, 1, 1};
static unsigned int hfuse_gridDim_30[] = {28672, 1, 1};
static unsigned int hfuse_blockDim_30[] = {192, 1, 1};
static unsigned int hfuse_gridDim_31[] = {229376, 1, 1};
static unsigned int hfuse_blockDim_31[] = {288, 1, 1};
static unsigned int hfuse_gridDim_32[] = {1858560, 1, 1};
static unsigned int hfuse_blockDim_32[] = {96, 1, 1};
static unsigned int hfuse_gridDim_33[] = {10913792, 1, 1};
static unsigned int hfuse_blockDim_33[] = {96, 1, 1};
static unsigned int hfuse_gridDim_34[] = {173056, 1, 1};
static unsigned int hfuse_blockDim_34[] = {288, 1, 1};
static unsigned int hfuse_gridDim_35[] = {173056, 1, 1};
static unsigned int hfuse_blockDim_35[] = {192, 1, 1};
static unsigned int hfuse_gridDim_36[] = {350208, 1, 1};
static unsigned int hfuse_blockDim_36[] = {224, 1, 1};
static unsigned int hfuse_gridDim_37[] = {173056, 1, 1};
static unsigned int hfuse_blockDim_37[] = {192, 1, 1};
static unsigned int hfuse_gridDim_38[] = {173056, 1, 1};
static unsigned int hfuse_blockDim_38[] = {192, 1, 1};
static unsigned int hfuse_gridDim_39[] = {229376, 1, 1};
static unsigned int hfuse_blockDim_39[] = {288, 1, 1};
static unsigned int hfuse_gridDim_40[] = {1858560, 1, 1};
static unsigned int hfuse_blockDim_40[] = {96, 1, 1};
static unsigned int hfuse_gridDim_41[] = {10913792, 1, 1};
static unsigned int hfuse_blockDim_41[] = {96, 1, 1};
static unsigned int hfuse_gridDim_42[] = {66560, 1, 1};
static unsigned int hfuse_blockDim_42[] = {288, 1, 1};
static unsigned int hfuse_gridDim_43[] = {106496, 1, 1};
static unsigned int hfuse_blockDim_43[] = {192, 1, 1};
static unsigned int hfuse_gridDim_44[] = {350208, 1, 1};
static unsigned int hfuse_blockDim_44[] = {224, 1, 1};
static unsigned int hfuse_gridDim_45[] = {66560, 1, 1};
static unsigned int hfuse_blockDim_45[] = {192, 1, 1};
static unsigned int hfuse_gridDim_46[] = {66560, 1, 1};
static unsigned int hfuse_blockDim_46[] = {192, 1, 1};
static unsigned int hfuse_gridDim_47[] = {229376, 1, 1};
static unsigned int hfuse_blockDim_47[] = {288, 1, 1};
static unsigned int hfuse_gridDim_48[] = {1858560, 1, 1};
static unsigned int hfuse_blockDim_48[] = {96, 1, 1};
static unsigned int hfuse_gridDim_49[] = {10913792, 1, 1};
static unsigned int hfuse_blockDim_49[] = {96, 1, 1};
static unsigned int hfuse_gridDim_50[] = {131072, 1, 1};
static unsigned int hfuse_blockDim_50[] = {288, 1, 1};
static unsigned int hfuse_gridDim_51[] = {131072, 1, 1};
static unsigned int hfuse_blockDim_51[] = {192, 1, 1};
static unsigned int hfuse_gridDim_52[] = {350208, 1, 1};
static unsigned int hfuse_blockDim_52[] = {224, 1, 1};
static unsigned int hfuse_gridDim_53[] = {131072, 1, 1};
static unsigned int hfuse_blockDim_53[] = {192, 1, 1};
static unsigned int hfuse_gridDim_54[] = {131072, 1, 1};
static unsigned int hfuse_blockDim_54[] = {192, 1, 1};
static unsigned int hfuse_gridDim_55[] = {229376, 1, 1};
static unsigned int hfuse_blockDim_55[] = {288, 1, 1};
static unsigned int hfuse_gridDim_56[] = {1858560, 1, 1};
static unsigned int hfuse_blockDim_56[] = {96, 1, 1};
static unsigned int hfuse_gridDim_57[] = {10913792, 1, 1};
static unsigned int hfuse_blockDim_57[] = {96, 1, 1};
static unsigned int hfuse_gridDim_58[] = {36864, 1, 1};
static unsigned int hfuse_blockDim_58[] = {288, 1, 1};
static unsigned int hfuse_gridDim_59[] = {106496, 1, 1};
static unsigned int hfuse_blockDim_59[] = {192, 1, 1};
static unsigned int hfuse_gridDim_60[] = {350208, 1, 1};
static unsigned int hfuse_blockDim_60[] = {224, 1, 1};
static unsigned int hfuse_gridDim_61[] = {28672, 1, 1};
static unsigned int hfuse_blockDim_61[] = {192, 1, 1};
static unsigned int hfuse_gridDim_62[] = {28672, 1, 1};
static unsigned int hfuse_blockDim_62[] = {192, 1, 1};
static unsigned int hfuse_gridDim_63[] = {229376, 1, 1};
static unsigned int hfuse_blockDim_63[] = {288, 1, 1};
//------------------------------------------------------------------------------------------
static unsigned int bfuse_gridDim_0[] = {1924096, 1, 1};
static unsigned int bfuse_blockDim_0[] = {64, 1, 1};
static unsigned int bfuse_gridDim_1[] = {10979328, 1, 1};
static unsigned int bfuse_blockDim_1[] = {64, 1, 1};
static unsigned int bfuse_gridDim_2[] = {102400, 1, 1};
static unsigned int bfuse_blockDim_2[] = {216, 1, 1};
static unsigned int bfuse_gridDim_3[] = {172032, 1, 1};
static unsigned int bfuse_blockDim_3[] = {104, 1, 1};
static unsigned int bfuse_gridDim_4[] = {415744, 1, 1};
static unsigned int bfuse_blockDim_4[] = {152, 1, 1};
static unsigned int bfuse_gridDim_5[] = {94208, 1, 1};
static unsigned int bfuse_blockDim_5[] = {112, 1, 1};
static unsigned int bfuse_gridDim_6[] = {94208, 1, 1};
static unsigned int bfuse_blockDim_6[] = {112, 1, 1};
static unsigned int bfuse_gridDim_7[] = {294912, 1, 1};
static unsigned int bfuse_blockDim_7[] = {224, 1, 1};
static unsigned int bfuse_gridDim_8[] = {1866752, 1, 1};
static unsigned int bfuse_blockDim_8[] = {64, 1, 1};
static unsigned int bfuse_gridDim_9[] = {10921984, 1, 1};
static unsigned int bfuse_blockDim_9[] = {64, 1, 1};
static unsigned int bfuse_gridDim_10[] = {45056, 1, 1};
static unsigned int bfuse_blockDim_10[] = {216, 1, 1};
static unsigned int bfuse_gridDim_11[] = {114688, 1, 1};
static unsigned int bfuse_blockDim_11[] = {104, 1, 1};
static unsigned int bfuse_gridDim_12[] = {358400, 1, 1};
static unsigned int bfuse_blockDim_12[] = {152, 1, 1};
static unsigned int bfuse_gridDim_13[] = {36864, 1, 1};
static unsigned int bfuse_blockDim_13[] = {112, 1, 1};
static unsigned int bfuse_gridDim_14[] = {36864, 1, 1};
static unsigned int bfuse_blockDim_14[] = {112, 1, 1};
static unsigned int bfuse_gridDim_15[] = {237568, 1, 1};
static unsigned int bfuse_blockDim_15[] = {224, 1, 1};
static unsigned int bfuse_gridDim_16[] = {1874944, 1, 1};
static unsigned int bfuse_blockDim_16[] = {64, 1, 1};
static unsigned int bfuse_gridDim_17[] = {10930176, 1, 1};
static unsigned int bfuse_blockDim_17[] = {64, 1, 1};
static unsigned int bfuse_gridDim_18[] = {53248, 1, 1};
static unsigned int bfuse_blockDim_18[] = {216, 1, 1};
static unsigned int bfuse_gridDim_19[] = {122880, 1, 1};
static unsigned int bfuse_blockDim_19[] = {104, 1, 1};
static unsigned int bfuse_gridDim_20[] = {366592, 1, 1};
static unsigned int bfuse_blockDim_20[] = {152, 1, 1};
static unsigned int bfuse_gridDim_21[] = {45056, 1, 1};
static unsigned int bfuse_blockDim_21[] = {112, 1, 1};
static unsigned int bfuse_gridDim_22[] = {45056, 1, 1};
static unsigned int bfuse_blockDim_22[] = {112, 1, 1};
static unsigned int bfuse_gridDim_23[] = {245760, 1, 1};
static unsigned int bfuse_blockDim_23[] = {224, 1, 1};
static unsigned int bfuse_gridDim_24[] = {1879040, 1, 1};
static unsigned int bfuse_blockDim_24[] = {64, 1, 1};
static unsigned int bfuse_gridDim_25[] = {10934272, 1, 1};
static unsigned int bfuse_blockDim_25[] = {64, 1, 1};
static unsigned int bfuse_gridDim_26[] = {57344, 1, 1};
static unsigned int bfuse_blockDim_26[] = {216, 1, 1};
static unsigned int bfuse_gridDim_27[] = {126976, 1, 1};
static unsigned int bfuse_blockDim_27[] = {104, 1, 1};
static unsigned int bfuse_gridDim_28[] = {370688, 1, 1};
static unsigned int bfuse_blockDim_28[] = {152, 1, 1};
static unsigned int bfuse_gridDim_29[] = {49152, 1, 1};
static unsigned int bfuse_blockDim_29[] = {112, 1, 1};
static unsigned int bfuse_gridDim_30[] = {49152, 1, 1};
static unsigned int bfuse_blockDim_30[] = {112, 1, 1};
static unsigned int bfuse_gridDim_31[] = {249856, 1, 1};
static unsigned int bfuse_blockDim_31[] = {224, 1, 1};
static unsigned int bfuse_gridDim_32[] = {2031616, 1, 1};
static unsigned int bfuse_blockDim_32[] = {64, 1, 1};
static unsigned int bfuse_gridDim_33[] = {11086848, 1, 1};
static unsigned int bfuse_blockDim_33[] = {64, 1, 1};
static unsigned int bfuse_gridDim_34[] = {209920, 1, 1};
static unsigned int bfuse_blockDim_34[] = {216, 1, 1};
static unsigned int bfuse_gridDim_35[] = {279552, 1, 1};
static unsigned int bfuse_blockDim_35[] = {104, 1, 1};
static unsigned int bfuse_gridDim_36[] = {523264, 1, 1};
static unsigned int bfuse_blockDim_36[] = {152, 1, 1};
static unsigned int bfuse_gridDim_37[] = {201728, 1, 1};
static unsigned int bfuse_blockDim_37[] = {112, 1, 1};
static unsigned int bfuse_gridDim_38[] = {201728, 1, 1};
static unsigned int bfuse_blockDim_38[] = {112, 1, 1};
static unsigned int bfuse_gridDim_39[] = {402432, 1, 1};
static unsigned int bfuse_blockDim_39[] = {224, 1, 1};
static unsigned int bfuse_gridDim_40[] = {1925120, 1, 1};
static unsigned int bfuse_blockDim_40[] = {64, 1, 1};
static unsigned int bfuse_gridDim_41[] = {10980352, 1, 1};
static unsigned int bfuse_blockDim_41[] = {64, 1, 1};
static unsigned int bfuse_gridDim_42[] = {103424, 1, 1};
static unsigned int bfuse_blockDim_42[] = {216, 1, 1};
static unsigned int bfuse_gridDim_43[] = {173056, 1, 1};
static unsigned int bfuse_blockDim_43[] = {104, 1, 1};
static unsigned int bfuse_gridDim_44[] = {416768, 1, 1};
static unsigned int bfuse_blockDim_44[] = {152, 1, 1};
static unsigned int bfuse_gridDim_45[] = {95232, 1, 1};
static unsigned int bfuse_blockDim_45[] = {112, 1, 1};
static unsigned int bfuse_gridDim_46[] = {95232, 1, 1};
static unsigned int bfuse_blockDim_46[] = {112, 1, 1};
static unsigned int bfuse_gridDim_47[] = {295936, 1, 1};
static unsigned int bfuse_blockDim_47[] = {224, 1, 1};
static unsigned int bfuse_gridDim_48[] = {1989632, 1, 1};
static unsigned int bfuse_blockDim_48[] = {64, 1, 1};
static unsigned int bfuse_gridDim_49[] = {11044864, 1, 1};
static unsigned int bfuse_blockDim_49[] = {64, 1, 1};
static unsigned int bfuse_gridDim_50[] = {167936, 1, 1};
static unsigned int bfuse_blockDim_50[] = {216, 1, 1};
static unsigned int bfuse_gridDim_51[] = {237568, 1, 1};
static unsigned int bfuse_blockDim_51[] = {104, 1, 1};
static unsigned int bfuse_gridDim_52[] = {481280, 1, 1};
static unsigned int bfuse_blockDim_52[] = {152, 1, 1};
static unsigned int bfuse_gridDim_53[] = {159744, 1, 1};
static unsigned int bfuse_blockDim_53[] = {112, 1, 1};
static unsigned int bfuse_gridDim_54[] = {159744, 1, 1};
static unsigned int bfuse_blockDim_54[] = {112, 1, 1};
static unsigned int bfuse_gridDim_55[] = {360448, 1, 1};
static unsigned int bfuse_blockDim_55[] = {224, 1, 1};
static unsigned int bfuse_gridDim_56[] = {1874944, 1, 1};
static unsigned int bfuse_blockDim_56[] = {64, 1, 1};
static unsigned int bfuse_gridDim_57[] = {10930176, 1, 1};
static unsigned int bfuse_blockDim_57[] = {64, 1, 1};
static unsigned int bfuse_gridDim_58[] = {53248, 1, 1};
static unsigned int bfuse_blockDim_58[] = {216, 1, 1};
static unsigned int bfuse_gridDim_59[] = {122880, 1, 1};
static unsigned int bfuse_blockDim_59[] = {104, 1, 1};
static unsigned int bfuse_gridDim_60[] = {366592, 1, 1};
static unsigned int bfuse_blockDim_60[] = {152, 1, 1};
static unsigned int bfuse_gridDim_61[] = {45056, 1, 1};
static unsigned int bfuse_blockDim_61[] = {112, 1, 1};
static unsigned int bfuse_gridDim_62[] = {45056, 1, 1};
static unsigned int bfuse_blockDim_62[] = {112, 1, 1};
static unsigned int bfuse_gridDim_63[] = {245760, 1, 1};
static unsigned int bfuse_blockDim_63[] = {224, 1, 1};
//------------------------------------------------------------------------------------------
extern "C" void conv2d_6(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_0(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_4(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_3(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_5(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_2(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_7(float* __restrict, float* __restrict, float* __restrict);
extern "C" void conv2d_1(float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1(float* __restrict, float* __restrict, float* __restrict);
//------------------------------------------------------------------------------------------
extern "C" void bgemm_5_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_2_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_3_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_7_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_6_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_1_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_5_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_4_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_0_fused_hfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
//------------------------------------------------------------------------------------------
extern "C" void bgemm_6_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_4_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_7_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_0_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_1_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_6_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_1_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_2_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_6_conv2d_3_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_7_conv2d_4_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_5_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_3_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_2_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
extern "C" void bgemm_5_conv2d_0_fused_bfuse(float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict, float* __restrict);
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
