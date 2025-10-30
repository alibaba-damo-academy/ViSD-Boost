# Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training

[Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Cao_Boosting_Vision_Semantic_Density_with_Anatomy_Normality_Modeling_for_Medical_ICCV_2025_paper.pdf) (ICCV2025)

## Inference and Evaluation
We provide the checkpoint trained on **CT-RATE**.
Download BiomedVLP-CXR-BERT-specialized from [link](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized) as used by CT-CLIP.

To perform inference, run:
```bash
python eval.py
```

To calculate evaluation metrics, run:
```bash
python calc_metrics.py
```

## Citation
If you find this repository useful, please cite:

```bibtex
@inproceedings{cao2025boosting,
  title={Boosting vision semantic density with anatomy normality modeling for medical vision-language pre-training},
  author={Cao, Weiwei and Zhang, Jianpeng and Shui, Zhongyi and Wang, Sinuo and Chen, Zeli and Li, Xi and Lu, Le and Ye, Xianghua and Zhang, Qi and Liang, Tingbo and Zhang Ling},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={23041--23050},
  year={2025}
}
