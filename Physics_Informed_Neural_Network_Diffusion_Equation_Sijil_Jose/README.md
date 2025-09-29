# PINNDE : Physics Informed Neural Networks for Diffusion Equation | GSoC 2025

![ML4Sci@GSoC2024](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*8KAp7eW2atsaRwdS.jpeg)

## Project Description :

The over arching goal of this project is to develop a proof of concept for building a fast and reliable sampler by solving reverse-time diffusion equation that leverages the high accuracy of diffusion models with the flexibility of physics-informed neural networks. PINNDE can be the basis of a fast, accurate, sampler of complicated and, or, intractable distributions in multiple dimensions. Encouraging results of the PINNDE method in 1, 2, and 3 dimensions are obtained.  

### What was Accomplished?
As part of GSoC 2025, I contributed to this project titled as 'PINNDE:Physics Informed Neural Networks for Diffusion Equation' with the organisation Machine Learnign for Science [ML4SCI](https://ml4sci.org/). I am working under the mentorship of Prof. Harrison Prosper, Prof. Pushpalatha Bhat, and Prof. Sergei Gleyzer. This project is part of the broader [GENIE](https://ml4sci.org/activities/gsoc2025.html) initiative within ML4SCI, which explores the use of machine learning techniques for anomaly detection and event generation in high-energy particle physics.
Our final goal is to test this method named PINNDE on toy examples and later move on to use it for devloping a fast simulations of particle jets. As part of GSoC 2025 i have finished the following tasks

- Implemented an accurate and stable approximant for q-function that is required for solving the reverse-time diffusion ODE.
- Implemented different PINN architechures to test the feasilibilty of PINNs to accurately solve the reverse-time diffusion ODE.
- Obtained satisfactory results on different probability distributions of 1 ,2 and 3 dimensions.
- Tested different optimisation strategies for traning PINNs
- Started using this new method on [Fast Calorimeter Challenge 2022 for benchmarking](https://calochallenge.github.io/homepage/) (coming Soon!!) 

Following are the relevant documents pertaining to this project. 

- Code on GENIE Github Repository: [Link to official Repository](https://github.com/ML4SCI/GENIE/tree/main/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose)
- Code on my Github Repository (my fork) : [Link to my fork (branch PINNDE)](https://github.com/sijil-jose/GENIE/blob/PINNDE/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/README.md)
- Project Documentation: (final blog coming soon !!)

#### Other Important Documents:
- Initial project idea from ML4SCI : [ML4SCI LinK](https://ml4sci.org/gsoc/2025/proposal_GENIE5.html)
- My project proposal : [Proposal](https://github.com/sijil-jose/GENIE/blob/PINNDE/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/slides_docs/GSOC_2025_Project_Proposal_Sijil_Jose.pdf)
- GSoC Abstract : [Abstract](https://summerofcode.withgoogle.com/programs/2025/projects/uGmyAV1q)
- Mid Term blog summarising the project : [PINNDE mid-term blog](https://medium.com/@sijiljose.999/gsoc-2025-with-ml4sci-part-i-physics-informed-neural-network-for-diffusion-equation-pinnde-491d46a5b84d)
- Final Document : (Coming Soon!!)
- Midterm Lighting Talk : [Midterm slides](https://github.com/sijil-jose/GENIE/blob/PINNDE/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/slides_docs/Mid-term_slides.pdf)

### Next Steps
- Finish implementing this method for Fast Calorimeter Challenge
- Explore Other PINN and operator learning frameworks.
- Add more unit tests for the files

### My Contributions:

Initally I had written a detailed [proposal](https://github.com/sijil-jose/GENIE/blob/PINNDE/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/slides_docs/GSOC_2025_Project_Proposal_Sijil_Jose.pdf) outlining my plans for the project and also finshed a [test task](https://github.com/sijil-jose/GENIE/tree/PINNDE/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/Initial_test). The following is the code developed during the GSoC 2025 coding period. 
- Code on GENIE Github Repository: [Link to official Repository](https://github.com/ML4SCI/GENIE/tree/main/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose)
- Code on my Github Repository (my fork) : [Link to my fork (branch PINNDE)](https://github.com/sijil-jose/GENIE/blob/PINNDE/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/README.md)

I had also worked on documenting my work in the form of blogs and stared compiling the results we obtained into an article, which can be found below.
- Mid Term blog summarising the project : [PINNDE mid-term blog](https://medium.com/@sijiljose.999/gsoc-2025-with-ml4sci-part-i-physics-informed-neural-network-for-diffusion-equation-pinnde-491d46a5b84d)
- Final Document : (Coming Soon!!)
- Preprint of the article: (Coming Soon !!)

### Description of Directories and files:
- ```flow_de``` : directory containing the files and scripts to train different models
  -   ``` flow_de.py``` : python file containg the classes named ```class qVectorField``` and ```class FlowDE``` for definig the q-function and numerically solving the reverse-time diffusion equation.
  -   ```gendata.py``` : python file continaing functions to sample from 1 , 2 and 3 dimensional distibutions considered in this project.
  -   ``` networks_1d.py``` , ``` networks_2d.py``` and ``` networks_3d.py``` : python files containing majority of the pytorch functions required to defining and training the neural networks for different cases.    
  -  ```train_1d_GMM.py```, ```train_2d_GMM.py```, ```train_3d_GMM.py``` : python files to train the PINNDE models. (uncomment the last line to run the optimiser )

- ```Jupyter Notebooks``` : contains the respective jupyter notebooks with more detailed explainations for each case
- ```FlowDE``` : contained ```FlowDE.ipynb``` a jupyternotebook with code to numerically solve the reverse-time diffusion equation for a 1D case.
- ```slides_docs``` : containes some pdf documents related to this project
- ```Tests``` : python files with unit tests for each functions. (To be updated)
- ```Figures```: contains some plots describing the results obtained in this project.
- ```README.md```: This documentation file
- ```Initial_test```: This directory contains all the files sumbitted as part of the initial tests as part of the application for GSoC 2025.


# Overview of interesting results obtained during this program (Plots):

### Plots comparing the target distributions and distributions obtained from the trained model:

![alt text](https://github.com/sijil-jose/GENIE/blob/PINNDE/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/Figures/fig_1Ddist.png)

these figures compares samples generated by the trained PINNDE network with those from the reference distribution used during training. The reference distribution is shown in black,
while samples from the PINNDE model are shown in blue. The two-dimensional and three-dimensional cases are visualized using corner plots: the diagonal panels display the marginalized one-dimensional distributions, while the off-diagonal panels illustrate the pairwise joint distributions
