# 1. Introduction

Large language models (LLMs) have demonstrated impressive performance in natural language processing (NLP) tasks, but they lack the ability to utilize visual cues for learning and reasoning about the real world. Additionally, they are unable to generate images, which is a crucial aspect of multimodal communication. In this paper, a novel approach called FROMAGE is proposed, which leverages a frozen LLM and a visual encoder to enable in-context learning for multimodal tasks.

FROMAGE uses a frozen large language model and a visual encoder. Both models can produce embeddings of different modalities separately, but they cannot interact with each other to explore multi-modal context. FROMAGE proposes a framework for creating an interaction layer between both models. The embeddings generated from the language model are mapped to the visual encoder's vector space using linear transformation to allow LLM to interact with the visual encoder. To generate an embedding for an image using the text input, they propose extending the vocabulary of LLM with the [RET] token. This represents an embedding of an image from the text input, which is learned during training to map the text space into the image space. They use the contrastive loss to learn the mapping. Moreover, with this approach, learning to generate the [RET] token for multi-modal dialogue emerges. Similar to text-visual embedding mapping, they also map the visual embedding to the text space using linear mapping. They use maximum likelihood estimation to learn the mapping.

The paper builds on the shortcomings and prior knowledge that previous work has provided. For example, Flamingo (Alayrac et al., 2022) proposed a visual language model for text generation; however, Flamingo cannot generate images. LIMBeR (Merullo et al., 2022) analyzes pretrained vision and language models and finds that learned representations are functionally equivalent up to a linear transform. Therefore, the authors propose to learn linear mappings for both modalities to be able to generate images while reasoning in a multi-modal setting.

The FROMAGE model is tested using multi-modal tasks, such as image retrieval and image captioning. The results provide evidence that this method is not only feasible for multi-modal learning, but it also maintains the existing capabilities of pre-trained text-only LLMs, such as in-context learning and greater sensitivity to input context. FROMAGE can use the additional descriptions to improve retrieval accuracy (9.0 to 10.4 on R@1). When prompted with the full multimodal context (i.e., 5 images and 4 stories), the model can learn in-context to synthesize plausible story-like text.

The potential of in-context learning piqued our interest, and we decided to explore prompting techniques to exploit the in-context learning capabilities of the FROMAGE model using few-shot learning. A. Beygelzimer et al. (2021) suggested that their work should be seen as a starting point or baseline for the area of research of multimodal few-shot learning. Therefore, we use this paper as a reference for our few-shot learning procedure. In this work, we aim to quantify to what extent the FROMAGE model can adapt to novel and out-of-domain tasks rapidly. Furthermore, we aim to verify that prompting with both visual and language information in few-shot learning is strictly more effective than prompting with language alone.

Our contributions are as follows:

Test the few-shot learning capabilities of the FROMAGE model in a multimodal setting.
Further analyze the multimodal vs. text-only few-shot learning capabilities of the FROMAGE model.
Compare the FROMAGE model to existing benchmark models and quantify the differences.

[In terms of your first contribution, testing the few-shot learning capabilities of FROMAGE in a multimodal setting, what specific tasks do you plan to evaluate the model on? Will you be using existing datasets or creating your own?

Regarding your second contribution, comparing the multimodal and text-only few-shot learning capabilities of FROMAGE, what do you expect to find? Do you anticipate that the model will perform better with multimodal prompts, or do you think that language prompts alone will be sufficient for few-shot learning?

Finally, for your third contribution, which benchmark models do you plan to compare FROMAGE against? What metrics will you use to evaluate performance?
]

# 2. Reproduction 

# 3. Extension

To test if the FROMAGe model can perform in-context learning we take inspiration from A. Beygelzimer et al. (2021) (Figure 4). We take their dataset - Open Ended Mini ImageNet. It consists of images and their corresponding captions. The interesting part is that the captions use nonsense words, for example 'dax' and 'blicket'. There are different few shot variants in the dataset. This means we can input 1, 3 or 5 images of dax and 1, 3 or 5 images of blicket in the prompt. There are 2500 samples for each few-shot variation and we take 500 random samples and try to retrieve a caption for the last image. We do this experiment to see if FROMAGe is able to perform in-context learning using a few-shot prompt. This did not work out-of-the-box and we thought it might have been because the language model cannot retrieve tokens that are not included in the vocabulary. For this reason, we add the new tokens (in this case 'dax' and 'blicket') to the vocabulary and perform the experiments again, which also did not help. We also experimented with shorter prompts (only 'dax' as prompt for instance), differnt nummber of words to be returned by the model, different temeperature parameters. Additionally, we experimented with 5 shots, which also did not improve the performance. (results examples to be added later)

If time allows, we will also experiment with the model retrieving an image (instead of a caption) of a 'dax' for example. 

# 4. REFERENCES:

@inproceedings{
tsimpoukelli2021multimodal,
title={Multimodal Few-Shot Learning with Frozen Language Models},
author={Maria Tsimpoukelli and Jacob Menick and Serkan Cabi and S. M. Ali Eslami and Oriol Vinyals and Felix Hill},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=WtmMyno9Tq2}
}




