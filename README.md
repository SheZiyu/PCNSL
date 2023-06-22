# PCNSL

This is code of paper:
Deep Learning-based Overall Survival (OS) Prediction Model in Patients with Rare Cancer: A Case Study for Primary Central Nervous System Lymphoma (PCNSL), published on International Journal of Computer Assisted Radiology and Surgery (2023).

Link of paper:
https://link.springer.com/article/10.1007/s11548-023-02886-2

Keypoints of paper:

1. Data Collection and Data Preprocessing
   
   ![1](https://github.com/SheZiyu/PCNSL/assets/98766434/7796ca24-6420-45ed-891a-56252a59c8da)

   Data Collection: T1-weighted (T1), T2-weighted (T2), post-contrast T1-weighted (T1Gd) and OS of these patients are collected. OS positive means OS more than 1 year and negative means less than 1 year. There are 30 positive cases and 26 negative cases.

   Step #1: Bias field correction, registration, skull stripping, intensity normalization and voxel resampling using 3D Slicer.

   Step #2: Data augmentation including background removal, elastic deformation, random spatial cropping, random rotation and resizing using MONAI.

2. 3D Residual Network (ResNet), Transfer Learning and Gradient-weighted Class Activation Mapping (Grad-Cam)
   
   ![scratch_](https://github.com/SheZiyu/PCNSL/assets/98766434/fbdc7f0a-8715-4170-8ee0-ffb1b4cba978)

   3D ResNet Architecture: Input 3D images, use convolution layers and pooling layers as encoder to extract features, finally, use a linear FC layer to output OS classification results.

   Transfer Learning: Pre-train 3D ResNet on BraTS2020 for OS classification of patients with glioblastoma (BraTS2020 consists of multi-modal preoperative images of 235 glioblastoma patients from 19 institutions with OS), then, fine-tune the 3D ResNet on PCNSL dataset for the target task OS classification of patients with PCNSL.

   Grad-cam: Output pattern of the 3D ResNet.

3. Result

   ![000](https://github.com/SheZiyu/PCNSL/assets/98766434/20e8bd45-8b73-4271-824a-c653c3d2cff8)

   Qutitative Result: Cross-validation results of ML models and 3D ResNet. SVM (Support Vector Machine). T (Training from Scratch). TL (Transfer Learning). Clinic means clinical data. T1r, T2r and T1Gdr mean radiomics data from T1, T2 and T1Gd. * means p < 0.05 in the t-test. Best performance on T1Gd, consistent with clinical outcome.

   ![tp](https://github.com/SheZiyu/PCNSL/assets/98766434/949a93f8-ccf5-467e-89c2-ddcef5504991)

   ![tn](https://github.com/SheZiyu/PCNSL/assets/98766434/2b305b82-9852-4da0-9abf-2823afa3344a)

   Qulitative Result: OS pattern of 3D ResNet. #number (patient ID). MRI (TIGd slice near tumor). Activation Map (activation map of the slice). Colorbar shows activation map intensity. PCNSL is a whole-brain disease; in cases where OS less than 1 year (Bottom), it is more difficult to distinguish tumor boundary from normal part of the brain than where OS more than 1 year (Top); consistent with clinical outcome. 

Citation:
@article{she2023deep,
  title={Deep learning-based overall survival prediction model in patients with rare cancer: a case study for primary central nervous system lymphoma},
  author={She, Ziyu and Marzullo, Aldo and Destito, Michela and Spadea, Maria Francesca and Leone, Riccardo and Anzalone, Nicoletta and Steffanoni, Sara and Erbella, Federico and Ferreri, Andr{\'e}s JM and Ferrigno, Giancarlo and others},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--8},
  year={2023},
  publisher={Springer}
}













