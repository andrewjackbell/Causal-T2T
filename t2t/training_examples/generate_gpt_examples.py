from os import path
import time
import openai
import pandas
import argparse
openai.api_key = "" #Put OpenAI API key here

def call_api(prompt):
        wait_time = 30
        while True:
            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}])

                response = completion.choices[0].message.content
                return response

            except openai.error.RateLimitError as e:
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                return call_api(prompt)

def create_dataset(proc_dir, example_count):

    with open(path.join(proc_dir,"encoded_reports.tsv"),'r') as f:
        df = pandas.read_csv(f,sep='\t',nrows=int(example_count))

    encoded_reports = df['report']
    generated_sentences = []

    for report in encoded_reports:
        gpt_prompt = f"Convert this encoded ADE report into a single concise sentence: ```{report}```"

        response = call_api(gpt_prompt)

        generated_sentences.append(response)


    with open("gpt_examples.tsv",'w+') as f:
        for i in range(len(encoded_reports)):
            report = encoded_reports[i]
            sentence = generated_sentences[i]
            f.write(report+"\t"+sentence+"\n")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--proc-dir', type=str)
    parser.add_argument('--example-count',type=str)

    args = parser.parse_args()

    create_dataset(args.proc_dir,args.example_count)

if __name__ == "__main__":
    main()