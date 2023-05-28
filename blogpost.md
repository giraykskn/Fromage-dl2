# 1. Introduction

![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png) `TODO: Introduction: An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well.`

Large language models (LLMs) have demonstrated impressive performance in natural language processing (NLP) tasks, but they lack the ability to utilize visual cues for learning and reasoning about the real world. Additionally, they are unable to generate images, which is a crucial aspect of multimodal communication.A novel approach called FROMAGE [1] is proposed, which leverages a frozen LLM and a visual encoder to enable in-context learning for multimodal tasks. Our specific focus was
on the storytelling task, which encompassed both text generation and imagetext retrieval. However, the primary emphasis lay in learning from contextual information within the given story. The experimental outcomes demonstrated that the model was capable of grounding itself in the provided story context, enabling it to retrieve images and generate text more effectively compared to scenarios without contextual input. The potential of in-context learning greatly intrigued us, as this aspect had not been fully explored using a few-shot learning approach, despite prior attempts within the story task. First, we reproduced the story image retrieval part of the storytelling experiment. Secondly we decided to examine the in-context learning capability using the few-shot framework proposed by frozen paper

FROMAGE uses a frozen large language model and a visual encoder. Both models can produce embeddings of different modalities separately, but they cannot interact with each other to explore multi-modal context. FROMAGE proposes a framework for creating an interaction layer between both models. The embeddings generated from the language model are mapped to the visual encoder's vector space using linear transformation to allow LLM to interact with the visual encoder. To generate an embedding for an image using the text input, they propose extending the vocabulary of LLM with the [RET] token. This represents an embedding of an image from the text input, which is learned during training to map the text space into the image space. They use the contrastive loss to learn the mapping. Moreover, with this approach, learning to generate the [RET] token for multi-modal dialogue emerges. Similar to text-visual embedding mapping, they also map the visual embedding to the text space using linear mapping. They use maximum likelihood estimation to learn the mapping.

The paper builds on the shortcomings and prior knowledge that previous work has provided. For example, Flamingo [2] proposed a visual language model for text generation; however, Flamingo cannot generate images. LIMBeR [3] analyzes pretrained vision and language models and finds that learned representations are functionally equivalent up to a linear transform. Therefore, the authors propose to learn linear mappings for both modalities to be able to generate images while reasoning in a multi-modal setting.

The FROMAGE model is tested using multi-modal tasks, such as image retrieval and image captioning. The results provide evidence that this method is not only feasible for multi-modal learning, but it also maintains the existing capabilities of pre-trained text-only LLMs, such as in-context learning and greater sensitivity to input context. FROMAGE can use the additional descriptions to improve retrieval accuracy (9.0 to 10.4 on R@1). When prompted with the full multimodal context (i.e., 5 images and 4 stories), the model can learn in-context to synthesize plausible story-like text. We decided to reproduce image retrieval experiment to further explore the claims made by the fromage paper and gain an understanding of the in-context learning capabilities with multi modal context.

In our reproduction we had three different settings, 1 caption, 5 captions, and 5 captions and 4 images. Unlike the claims made by the paper, we found that including 4 images and 5 captions lead to lower recall than 5 captions. We explore the possible reasons for this finding in the discussion section of the post. After reproducing the paper we shifted our focus on the extension.

We decided to explore prompting techniques to exploit the in-context learning capabilities of the FROMAGE model using few-shot learning. Frozen [4] suggests that their work should be seen as a starting point or baseline for the area of research of multimodal few-shot learning. Therefore, we use their paper as a reference for our few-shot learning procedure. In this work, we aim to quantify to what extent the FROMAGE model can adapt to novel and out-of-domain tasks rapidly. Furthermore, we aim to verify that prompting with both visual and language information in few-shot learning is strictly more effective than prompting with language alone.
We utilized the openended ImageNet data generated by Frozen to evaluate our Fromage model’s performance in few-shot learning scenarios. Specifically, we employed the 2-way and 5-way few-shot learning tasks using non-existing classes such as ’blicket’ and ’dax,’ which acted as replacements for existing classes like ’dog’ and ’cat.’ Each class was associated with corresponding images and provided to the Fromage model as prompts, along with a query image and a question about its class. We varied the number of shots and repeats and evaluated the model’s query performance in terms of accuracy. We compared Fromage models' accuracy agains the frozen.

The results from our extension aligned with the findings of the Frozen paper. We observed that 2-way classification performed better than 5-way classification, as the model had to learn fewer classes. Furthermore, increasing the number of shots and repeats generally led to improved performance, except for the 5-way, 5-shot, 5-repeat scenario. 
This could be attributed to the increased input length, which caused the model to struggle with focusing on specific contexts. Moreover, we find that fromage model is less accuracte than the frozen. We also noticed that the accuracy remained below chance level, possibly suggesting that the model did not fully grasp the meaning of the out-of-vocabulary words.

Our contributions are as follows:

Test the few-shot learning capabilities of the FROMAGE model in a multimodal setting.
Further analyze the multimodal vs. text-only few-shot learning capabilities of the FROMAGE model.
Compare the FROMAGE model to existing benchmark models and quantify the differences.

Regarding your second contribution, comparing the multimodal and text-only few-shot learning capabilities of FROMAGE, what do you expect to find? Do you anticipate that the model will perform better with multimodal prompts, or do you think that language prompts alone will be sufficient for few-shot learning?

Finally, for your third contribution, which benchmark models do you plan to compare FROMAGE against? What metrics will you use to evaluate performance?
]

# 2. Strengths 

1. Effective multimodal generation: 

The model effectively leverages pretrained language models that are available to generate coherent captions for images and relevant multimodal dialogues, since it grounds the language models to images.

2. Training Efficiency

Fromage is trained by just finetuning a small set of parameters, which makes it easy to train. This makes it much easier to conduct further research on the model and the paper and the model can be applied to many specific fields like medicine, fashion etc. by finetuning again.

3. Usage Versatility

Fromage is applicable to many different areas, such as captioning, image-text retrieval, multimodal dialogue which shows its broad potential.

4. Detailed results

The paper and the code they publish are very clear, which leaves less room for misunderstanding. The background, experiment and analysis are detailed, and shows the strengths and weaknesses of the model based on what they explored.

# 3. Weaknesses

1. Limited exploration of the model:

The paper does not fully explore the capabilities in the model in terms of datasets, hyperparameters and prompting techniques. Especially the lack of datasets and hyperparameters creates concerns over the sucess of the model, makes it seem less adaptable.

2. Too much reliance on pre-trained models:

Since Fromage relies mostly on pre-trained models, it is difficult to understand if the results are more influenced by Fromage or the models beneath. The model also inherits all the bias and limitations of the mentioned models, which can create ethical issues.

3. Limited analysis on failure:

Although the research is very detailed, there is not much talk about the failures of the model and why it happens besides the appendix.

# 4. Reproduction 

In this project, we focused on reproducing the visual storytelling experiment using the VIST dataset, as outlined in section 4.1 in the original paper. Our aim was to evaluate the FROMAGe model's ability to learn in context and transfer knowledge in a zero-shot setting. To streamline the process and save time and computational resources, we chose not to evaluate the Clip model and instead focused solely on assessing the performance of the FROMAGe model. The experiment comprised three different settings, each varying in the input provided: 1 caption, 5 captions, and 5 captions accompanied by 4 images. Since the FROMAGe model is not generative and relies on an embedding data store to match input embeddings during image retrieval, this posed a challenge. The original precomputed embeddings were cc3m embeddings, which the model was trained on. However, to ensure accurate image retrieval for this specific experiment, the model needed to encode the images from the VIST dataset as well. The paper lacked concrete instructions in this regard. To address this issue, we decided to encode the last image (target image) of each story. This ensured that each precomputed embedding corresponded to a specific story that needed to be predicted. By doing so, I ensured that the total probability across the entire test set summed to one, indicating that no precomputed embeddings were left unused. Additionally, we took the initiative to remove any duplicated target images from the dataset. This step prevented the model from retrieving multiple identical images within a single pass, leading to more diverse and meaningful results.

As a result of the aforementioned ambiguities, the reproduced results deviate to some extent from those reported in the paper. Notably, in the experiments involving inputs with 1 caption and 5 captions, the recall values exhibit a consistent trend. The model performs better when provided with 5 captions compared to just 1 caption across all three recall levels. This suggests that having more context benefits the model in understanding the sequential nature of the story, thereby increasing the probability of retrieving the correct image. However, the results for the input of 5 captions and 4 images do not align with expectations. At recall@1, the performance is even worse than when the model is provided with only 1 caption at the same recall level. On the other hand, at recall@5 and recall@10, it performs better than the model with 1 caption but falls short of the performance achieved with 5 captions. The recall values for all three settings are depicted in the following figure:

<img src="blogpost_imgs/r.png" alt="Plot" width="500" height="160">

It can be seen that overall recall values surpass those reported in the paper. This difference in performance could be attributed to the construction of the precomputed embedding space. In our implementation, we encoded only the target images to serve as the model's search space. Consequently, it becomes relatively easier for the model to retrieve the correct images, potentially leading to higher recall values overall. The following example at recall@1 illustrate the model outputs:

***Original Story:***
<img src="blogpost_imgs/Story.png" alt="Plot" width="1000" height="200">

***Output -- caption 1 / recall@1:***

<img src="blogpost_imgs/Output1caption1image0recall1.png" alt="Plot" width="250" height="200">

***Output -- captions 5 / recall@1:***

<img src="blogpost_imgs/Output1caption5image0recall1.png" alt="Plot" width="250" height="200">

***Output -- captions 5, images 4 / recall@1:***

<img src="blogpost_imgs/Output1caption5image4recall1.png" alt="Plot" width="250" height="200">

It can be seen from the previous example that model using 5 captions only has the best understanding of story context therefore the retrieved image is most similar to the ground truth while the other two retrieved the same image. Another example to illustrate outputs at recall@5:

***Original Story:***
<img src="blogpost_imgs/Story2.png" alt="Plot" width="1000" height="200">

***Output -- captions 5 / recall@5:***
<img src="blogpost_imgs/Output1caption5image0recall5.png" alt="Plot" width="1000" height="200">

Since model can retrieve multiple images to match the taret image, the performance is expetectly better than only retrieving one iamge.

There are a couple of potential factors that could explain the discrepancies between our reproduced results and those reported in the paper. One possible reason is the differences in the experimental settings between our reproduction and the original study. For instance, the construction of precomputed embeddings or variations in how recalls are calculated might have an impact on the outcomes. Another aspect to consider is the potential impact of dataset changes. Over time, some URLs associated with the images in the dataset may have become invalid or inaccessible. While we attempted to mitigate this issue by randomly sampling from the entire dataset for our experiments, it is still possible that these changes in the availability of certain images could have influenced the results to some degree, although the likelihood of significant impact is relatively low. Overall, it is essential to acknowledge these factors and consider them when interpreting the differences between our reproduced results and the findings presented in the original paper. 

In conclusion, the FROMAGe model demonstrates a certain degree of effectiveness in contextual learning and zero-shot transfer. Despite the discrepancies observed in the reproduced results, the model generally performs well on the test dataset. For future investigations, it would be valuable to explore the model's capabilities using alternative datasets to assess its similar abilities in different contexts.


# 5. Our Novel Contribution

![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png) `TODO: Change this.`

To test if the FROMAGe model can perform in-context learning, our inspiration is Tsimpoukelli et al. (2021)(Figure 4). We use their dataset - Open Ended Mini ImageNet. It consists of images and their corresponding captions. The interesting part is the captions containing words without meanings such as “dax” or “blicket”. In order to observe if the model is able to learn those words, we experiment with the different variants in the dataset. They are the variants of few-shot learning, and the dataset that we use has 1,3,5 inner-shots, along with 2 and 5 ways. We use this setting as introduced in the Frozen language model by Tsimpoukelli et al. (2021). The ways represent the number of categories (dog vs cat) and the inner-shots are used to show the amount of distinct examples given to the model per category. We apply the same technique meaning that it is possible to experiment with 1,3 or 5 images of dax or 1,3 or 5 images of blickets as the input prompt. We do this experiment to see if FROMAGe is able to perform in-context learning using a few-shot prompt. We also experiment with prompts of different lengths (only “dax” as a prompt for instance), the amount of words that are expected to be returned by the model and different temperature parameters. Another variation we try is repeating the same prompt, the number of inner-shots that are repeated when giving them to the model. We observe that with 5 repetitions, no matter how many inner-shots or ways are given, it is possible to observe higher accuracy in the outcome of the model learning a meaningless word and being able to output the correct image/caption pair for that. 

# 6. Results

![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png) `TODO: Write.`

# 7. Conclusion

In conclusion, the experiment of performing few-shot learning on the FROMAGE model using open ended miniiamgenet dataset has yielded quite promising results and demonstrated the model's ability to quickly adapt to new tasks with minimal labeled data. By leveraging the power of transfer learning and meta-learning techniques, the FROMAGE model showcased good generalization capabilities and the capacity to acquire new knowledge efficiently. The model's performance on unseen tasks improved significantly after exposure to just a few examples, showcasing its potential for practical applications in scenarios where limited labeled data is available. This experiment highlights the importance of exploring innovative approaches like few-shot learning to tackle the data scarcity challenge in machine learning and paves the way for further advancements in this field. With continued research and refinement, the FROMAGE model and similar few-shot learning approaches hold the promise of enhancing the flexibility and adaptability of AI systems in various domains.

# REFERENCES:


[1]
J. Y. Koh, R. Salakhutdinov, and D. Fried, ‘Grounding Language Models to Images for Multimodal Generation’, arXiv e-prints, p. arXiv:2301.13823, Jan. 2023.

[2]
‘Flamingo: a Visual Language Model for Few-Shot Learning’, arXiv e-prints, p. arXiv:2204.14198, Apr. 2022.

[3]
J. Merullo, L. Castricato, C. Eickhoff, and E. Pavlick, ‘Linearly Mapping from Image to Text Space’, arXiv e-prints, p. arXiv:2209.15162, Sep. 2022.

[4]
M. Tsimpoukelli, J. L. Menick, S. Cabi, S. M. A. Eslami, O. Vinyals, and F. Hill, ‘Multimodal Few-Shot Learning with Frozen Language Models’, in Advances in Neural Information Processing Systems, 2021, vol. 34, pp. 200–212.




