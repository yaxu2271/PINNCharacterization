# Deep Learning for Full-Field Ultrasonic Characterization

## Overview
This repository contains the code used in the paper "**Deep Learning for Full-Field Ultrasonic Characterization**". DOI: https://doi.org/10.1016/j.ymssp.2023.110668. The code includes implementations for direct inversion methods and Physics-Informed Neural Networks (PINNs), aimed at reconstructing the mechanical properties of layered components from ultrasonic testing data. These models provide a robust, data-driven framework for nondestructive evaluation of complex materials.

### Key Contributions:
- **Direct Inversion Method**: A systematic approach using spectral denoising and differentiation of full-field ultrasonic data to build appropriate neural maps for reconstructing unknown material properties.
- **Physics-Informed Neural Networks (PINNs)**: An implementation to learn the physical parameters and predict wave propagation in complex systems while leveraging physics constraints during training.
- **Application to Ultrasonic Testing**: These models are applied to reconstruct material properties from ultrasonic datasets, which have applications in areas like medical imaging and material quality assessment.

## Repository Structure
- `data/`: Sample ultrasonic datasets for testing the models.
- `direct_inversion/`: Implementation of the direct inversion method.
- `pinn/`: Implementation of the Physics-Informed Neural Networks.
- `results/`: Generated figures and validation results for both synthetic and experimental data.

## Run
To run the code, you will need Python 3.7 or higher, PyTorch 2.2.2 or higher. 


## Requirements
- Python >= 3.7
- PyTorch >= 2.2.2
- NumPy
- SciPy
- Matplotlib
- scikit-learn

## Citation
If you find this code useful for your research, please consider citing our paper:

```
@article{xu2023deep,
  title={Deep learning for full-field ultrasonic characterization},
  author={Xu, Yang and Pourahmadian, Fatemeh and Song, Jian and Wang, Conglin},
  journal={Mechanical Systems and Signal Processing},
  volume={201},
  pages={110668},
  year={2023},
  publisher={Elsevier}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact
For any questions or collaboration inquiries, please feel free to reach out at:

- Yang Xu: [Yang.Xu@colorado.edu]
