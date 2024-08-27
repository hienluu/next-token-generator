import gradio as gr

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string

from large_model import LanguageModel

MODEL_NAME = "gpt2"

myLanguageModel = LanguageModel(MODEL_NAME)

def quick_test(prompt):
    myLanguageModel = LanguageModel("gpt2")
    next_token_probabilities = myLanguageModel.getNextTokenProbabilities(prompt, 10)
    update_next_token_probabilities = {k: round(v * 100, 2) for k, v in next_token_probabilities.items()}
    print("next_token_probabilities type: ", type(next_token_probabilities))


# The beginning of the Gradio app
prompt_examples = ["The sky is", 
                   "The capital of France is", 
                   "Earth and moon",
                   "Once upon a",
                   "The quick brown fox jumps over the"
                   ]

def plot_pie_chart(data):    
    labels = list(data.keys())
    sizes = list(data.values())
    
    fig, ax = plt.subplots()    
    #ax.pie(sizes, labels=labels)
    ax.pie(sizes, labels=labels, autopct='%1.2f%%', normalize=False)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.    
    return fig

def plot_next_token_probabilities(prompt):
    next_token_probabilities = myLanguageModel.getNextTokenProbabilities(prompt, 10)
    updated_next_token_probabilities = {k: round(v * 100, 2) for k, v in next_token_probabilities.items()}
    #output_per_line = "\n".join(f"{k}: {v}" for k, v in update_next_token_probabilities.items())

    updated_next_token_probabilities_table = {
        "Token": list(updated_next_token_probabilities.keys()),
        "Probability (%)": list(updated_next_token_probabilities.values())
    }

    updated_next_token_probabilities_df = pd.DataFrame.from_dict(updated_next_token_probabilities_table)
    return plot_pie_chart(next_token_probabilities), updated_next_token_probabilities_df


def update_prompt(prompt, evt: gr.SelectData):
    if evt.value in string.punctuation:
      return f"{prompt}{evt.value}"
    else:
      return f"{prompt} {evt.value}"

def clear_prompt(prompt):
   return ""

# color palettes https://www.color-hex.com/color-palette/23362
css = """
  .gradio-container { height: 500vh !important; background-color: #E6F3FF; height: 100%}
  #title { color: #414931; text-align: center}
  #generate_btn {background: #ff6600}
  #clear_btn {background: #bad1b1;}
  #prompt {background: #D4EDDA;}
  #examples_btns {background: #FEFCBF}
  #examples_btns button {color: #C05621; border: 2px solid #14142e}
  #plob_plot label {color: #0e3768; font-size: 1.1em}
  #prob_table p { color: #0e3768; font-size: 1.1em}
  #prob_table th {background-color: #0e3768; color: white}
  #prob_table td {background-color: #d5ddf3}
"""
demo = gr.Blocks(css=css, fill_width=True, fill_height=True)

with demo:
  gr.Markdown(elem_id="title", value="# GPT-2 Next Token Probabilities Demo")
  with gr.Row(equal_height=False):
      with gr.Column(scale=5):
        prompt = gr.Textbox(elem_id="prompt", label="Prompt", lines=1, scale=5)
        with gr.Row():
            generate_btn = gr.Button(elem_id="generate_btn", value="Generate", scale=2)
            clear_btn = gr.Button(elem_id="clear_btn", value="Clear", scale=1)
      with gr.Column(scale=5):
        examples = gr.Examples(elem_id="examples_btns", examples=prompt_examples, inputs=prompt)    
  with gr.Row(equal_height=False, ):
    with gr.Column(scale=5):
      prob_plot = gr.Plot(elem_id="plob_plot",label="Next Token Probabilities (Relative)")
    with gr.Column(scale=5):
        prob_df = gr.DataFrame(elem_id="prob_table", label="Next Token Probabilities (Raw)", headers=["Token", "Probability (%)"])
    
  prob_df.select(update_prompt, [prompt], [prompt])
  generate_btn.click(plot_next_token_probabilities, inputs=prompt, outputs=[prob_plot, prob_df])
  clear_btn.click(clear_prompt,[prompt], [prompt])

# main
if __name__ == "__main__": 
    gr.close_all()
    demo.launch(debug=True)