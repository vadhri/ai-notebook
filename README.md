# gen-ai-notebook

Data used for training. 
Great Expectations book from project gutenberg.

This repo is a mix of some guidance from various sources but primarily the two below. 
- Decode only LLM with details in paper below. 
https://arxiv.org/abs/1706.03762

- Code walk through using pytorch for the paper above.
https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3083s

Running the main python file will require a >= 15GB gpu to work preferably T4 or above. 
TODO: The training loss is significantly less but the val loss did not come down which hints of overfitting.