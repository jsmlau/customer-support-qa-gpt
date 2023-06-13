from langchain import PromptTemplate


class Prompt:
    def __init__(self):
        self.template = '''
        You are a helpful customer support representative and your task is answering customers' query.
        Only use information from the previous questions and answers to help you answer the customer's question.

        Previous conversation history:
        {history}

        Query from customer:
        {question}

        Previous questions and answers that may be helpful to you:
        {references}

        '''

    def get_prompt(self):
        prompt = PromptTemplate(
            input_variables=["history", "question", "references"],
            template=self.template,
        )
        prompt.save('data/prompt')

        return prompt

    def get_prompt_string(self, history, question, references):
        template = self.get_prompt()

        return template.format(
            history=history, question=question, references=references)
