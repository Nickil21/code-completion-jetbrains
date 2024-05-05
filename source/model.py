import pandas as pd


def reformat_input(data):
    data['input'] = data.apply(lambda x:'''
                                        {0}
                                        """
                                        {1}
                                        """
                                        '''.format(x['signature'], x['docstring'][:200]), axis=1)
    return data


valid_pandas = pd.read_json("temp/kotlin/valid_dataset.json", lines=True, orient="records")
valid_pandas = reformat_input(valid_pandas)

valid_input = valid_pandas["input"].tolist()[:2]
valid_body = valid_pandas["body"].tolist()[:2]


def get_prompt(lang, input_code):
    if lang == "kotlin":
        few_shot_prompts = '''
                Complete the code written in Kotlin. Write only the Answer.

                Question:
                {0}

                Answer:
                {1}

                Question:
                {2}

                Answer:
                {3}
                '''.format(valid_input[0], valid_body[0], valid_input[1], valid_body[1])
        given_prompt = '''
        Question:
        {0}

        Answer:

        '''.format(input_code)
        # print(few_shot_prompts)
        messages = few_shot_prompts + given_prompt
    if lang == "python":
        messages = input_code
    return messages



def get_finetuning_prompt(data_point):
    messages = '''
            [INST] <<SYS>>
            Complete the code written in Kotlin. Don't explain the code, just generate the code block itself as one continuous string.
            <</SYS>>

            [INST] {0} [/INST]
            {1}
            '''.format(data_point['input'], data_point['body'])
    return messages
