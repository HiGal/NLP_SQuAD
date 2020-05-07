import json
import pandas as pd

class Squad:
    def __init__(self, input_location):
        self.location = input_location
        file = open(input_location)
        json_file = json.load(file)
        self.version = json_file['version']
        self.data = json_file['data']
        
        df_builder = []
        for sample in self.data:
            title = sample['title']
            paragraphs = sample['paragraphs']
            
            for paragraph in paragraphs:
                context = paragraph['context']
                questions = paragraph['qas']
                
                for question in questions:
                    q_id = question['id']
                    q_content = question['question']
                    answers = question['answers']
                    is_impossible = question['is_impossible']
                    qas = {
                        'id':q_id,
                        'wiki_title':title,
                        'context':context,
                        'content':q_content,
                        'is_impossible':is_impossible
                    }
                    if is_impossible:
                        qas['answer'] = ""
                        qas['answer_start'] = -1
                    else:
                        answer = answers[0]
                        qas['answer'] = answer['text']
                        qas['answer_start'] = answer['answer_start']
                    df_builder.append(qas)
        self.df = pd.DataFrame(df_builder)
    
    def get_dataframe(self):
        return self.df

