# We define module for annotating a sentence with one entity. It can be reused for many entities, provided a definition.

from dspy.primitives.assertions import assert_transform_module, backtrack_handler
import dspy
from dspy import Suggest
import ast
from tqdm import tqdm
import openai
import pandas as pd
from dspy.teleprompt import LabeledFewShot
import pandas as pd
from pprint import pprint
import argparse

# This module is used to annotate a sentence with one entity, using a language model to analyze the sentence and return a list of NER labels for each word in the sentence.

class NERIndividualAnnotationModule(dspy.Module):
    """
    Module for annotating a sentence with one entity.
    This module can be reused for many entities, provided a definition.
    It uses a language model to analyze the sentence and return a list of NER labels for each word in the sentence.
    The labels are in BIO format:
        B- prefix indicates the beginning of an entity.
        I- prefix indicates the inside of an entity.
        O indicates a token outside any entity.
    """

    class NERIndividualAnnotationSignature(dspy.Signature):
        """
        Your job is to take a news article, and return a list of NER labels (one for each word in the article).

        Analyze each 1-gram, 2-gram and 3-grams of an entire article for NER labelling of your given entity. If any of the words in th n-grams in the article contain this given part-of-speech, it should be labeled.
        Identify where, if at all, the entity should be labeled on the sentence, using BIO format:
            B- prefix indicates the beginning of an entity.
            I- prefix indicates the inside of an entity.
            O indicates a token outside any entity.
            Every word should have exactly one entity tag.
            BIO format example: input string: "Word1 word2 word3", Reasoning: "one sentence analysis of word1, one sentence analysis of word2, one sentence analysis of word3" -> annotations: ["B-ABC", "I-ABC", "O"].
        If the given entity is not relevant to the n-gram or sentence in question, it should be labeled with "O".

        Also check that all entities are continuous (an O tag cannot be followed by an I tag).
        """
        # entity info
        given_entity_tag = dspy.InputField(desc="The given entity for NER labeling.")
        given_entity_description = dspy.InputField(desc="A description of how/when a word should be labeled with the given entity.")
        # input info
        input_str = dspy.InputField(desc="The text string to analyze for the presence of the given entity.")
        str_len= dspy.InputField(desc="The number of words in the input string, aka the number of tags in the output list")
        rationale = dspy.OutputField(desc="The resasoning behind each entity assigned to each word.")
        # outputs
        annotations = dspy.OutputField(desc="List of labels of the entities in the input string. Entity bank: 'B-BIAS', 'I-BIAS', 'O'. Format this as a string: ['B-BIAS', 'I-BIAS', 'O'] ")

    def __init__(self, entity_tag, entity_description):
        self.annotate = dspy.Predict(self.NERIndividualAnnotationSignature)
        self.entity_tag = entity_tag
        self.entity_description = entity_description

    def forward(self, input_str):
        input_str_len = len(input_str.split())

        annotation_pred_obj = self.annotate(
            given_entity_tag=self.entity_tag,
            given_entity_description=self.entity_description,
            input_str=input_str,
            str_len=str(input_str_len)
            )
        print("\n ======= Annotation prediction object: ======== \n\n")
        pprint(f"{annotation_pred_obj=}")
        print("\n ========================================================= \n\n")

        annotations_list = self._extract_list_from_str(annotation_pred_obj.annotations)

        Suggest(len(annotations_list) == input_str_len,
                f"The length of the annotation list should be {input_str_len} but it is {len(annotations_list)}."
                )

        return annotations_list, annotation_pred_obj

    def _extract_list_from_str(self, string):
        # slice out the list in string form
        first_bracket_pos = string.find('[')
        last_bracket_pos = string.rfind(']') + 1
        annotations_str = string[first_bracket_pos:last_bracket_pos]
        print("\n ======= Extracted list: ======== \n\n")
        pprint(f"{annotations_str=}")
        print("\n ========================================================= \n\n")
        try:
            annotations_list = ast.literal_eval(str(annotations_str))
        except Exception as e:
            print("Error extracting list from string:", string, e)
            annotations_list = annotations_str

        return annotations_list

 # Functions to Run Annotation + Aggregate Annotations
def aggregate_annotations_multi_label(annotations_list):
    """
    Aggregate multiple lists of NER annotations into a single list.
    Args:
        annotations_list (list of list): A list containing multiple lists of NER annotations.
    Returns:
        list: A single list containing aggregated NER annotations.
    """
    # Find the maximum length among all annotation lists
    max_len = max(len(annotations) for annotations in annotations_list)

    # Initialize the merged_annotations list with 'O' for the maximum length
    merged_annotations = [['O'] for _ in range(max_len)]

    # Iterate over each annotation list
    for entity_annotations in annotations_list:
        for i in range(len(entity_annotations)):
            if entity_annotations[i] != 'O':
                if merged_annotations[i] == ['O']:
                    merged_annotations[i] = [entity_annotations[i]]
                else:
                    merged_annotations[i].append(entity_annotations[i])

    return merged_annotations

def annotate_ner_multi_label(input_str):
    """
    Annotate the input string with NER labels for the given entity.
    Args:
        input_str (str): The input string to annotate.
    Returns:
        dict: A dictionary containing the input string, NER tags, rationale, and individual annotations.
    """
    # annotate with an agent for each entity
    annotations_list, pred_obj = compiled_annotator_agent(input_str=input_str)

    rationales_list = pred_obj['rationale']
    all_pred_objs = [pred_obj]

    annotations_list = []
    annotations_list.append(annotations_list)
    # Can add further annotations list later

    # return dict
    annotation = {}
    annotation['input_str'] = input_str
    annotation['ner_tags'] = aggregate_annotations_multi_label(annotations_list)
    annotation['rationale'] = str(rationales_list)
    annotation['individual_annotations'] = all_pred_objs

    return annotation


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run NER Individual Annotation Module")
    parser.add_argument('--openai_api_key', type=str, required=True, help='OpenAI API key')
    args = parser.parse_args()


    openai.api_key = args.openai_api_key #@param

    # Setup LM
    open_ai_model_endpoint = "gpt-4o" #@param
    default_model = dspy.OpenAI(model=open_ai_model_endpoint, 
                                max_tokens=2048, 
                                model_type="chat", 
                                temperature=0.8)

    default_model.drop_prompt_from_output = True
    dspy.settings.configure(lm=default_model, trace=[], temperature=0.8)
    print("Language model configured successfully.")

        
    entity_tag = "B-BIAS/I-BIAS"
    entity_description = (
            "Explicit or implicit bias can be identified by examining the following elements:\n\n"
            "- Loaded Language: Pay attention to words with strong emotional connotations, such as \"terrorist,\" \"illegal alien,\" or \"welfare queen.\".\n"
            "- Exaggeration and Hyperbole: Makes unsupported claims or presents ordinary situations as extraordinary to manipulate perceptions.\n"
            "- Stereotyping:Look for generalizations about entire groups of people, often based on race, ethnicity, gender, or religion, such as: \'my whole family\', \'all Muslims\', \'Women in general\', \'everyone in this city\', \'all politicians\',  \'tall people\', \'women\', \'the rocketry club\', \'students\', \'tall boys\'.\n"
            "- Slanting: This involves presenting information in a way that favors one side of an issue.\n"
            "- Manipulative Word Choices: Uses emotionally charged or misleading terms to subtly influence opinions.\n"
            "- Target Audience: Understand who the intended audience is and how that might influence the content.\n"
            "- Lack of Verifiable Sources: Relies on unverifiable or fabricated sources to support claims.\n"
            "- Logical Fallacies: Engages in flawed reasoning (e.g., circular logic, personal attacks) that undermines logical debate.\n"
            "- Selective Reporting: Notice if the article focuses on certain aspects of a story while ignoring others.\n"
            "- Inconsistencies and Factual Errors: Contains contradictions or easily disprovable inaccuracies, showing disregard for truth.\n"
            "- Selective Omission: Deliberately leaves out crucial information that would provide a more balanced understanding.\n\n"
            "Note that the length of the returned "
    )

    annotator_agent = NERIndividualAnnotationModule(
        entity_tag=entity_tag,
        entity_description=entity_description,
        )


    examples = {
        "input_str": [
            "Women are all such bad drivers",
            "Everyone from the city is rude and unfriendly",
            "John is usually late to meetings",
            "The college students on my street are always so loud"
        ],
        "rationale": [
            "The use of the word 'women' in this broad statement is a generalization because it is not about any specific woman or group of women. The use of the words 'are' and 'all' is a ubiquitous classification, extending the biased entity another two words.",
            "The phrase 'Everyone from the city' generalizes the behavior of all people from the city, which is a broad and unjustified assumption. The words 'is' and 'rude and unfriendly' extend this bias.",
            "This phrase makes a statement about John's punctuality, but does it is specific to one person, and uses the word 'usually' which is a hedge and not biased.",
            "This phrase unfairly characterizes a large group of people (country people) as loud. The use of the word 'always' is a ubiquitious classifier, therefore this contains a bias."
        ],
        "annotations": [
            "['B-BIAS', 'I-BIAS', 'I-BIAS', 'O', 'O', 'O']",
            "['B-BIAS', 'I-BIAS', 'I-BIAS', 'I-BIAS', 'O', 'O', 'O']",
            "['O', 'O', 'O', 'O', 'O', 'O', 'O']",
            "['O', 'B-BIAS', 'I-BIAS', 'I- BIAS', 'I-BIAS', 'I-BIAS', 'I-BIAS', 'O', 'O']"
        ]
    }
    df = pd.DataFrame(examples)
    dataset = []
    for input_str, rationale, annotations in df.values:
        input_str_len = str(len(input_str.split(" ")))
        dataset.append(dspy.Example(given_entity_tag=entity_tag, 
                                    given_entity_description=entity_description, 
                                    input_str=input_str, 
                                    str_len=input_str_len, 
                                    rationale=rationale, 
                                    annotations=annotations).with_inputs("given_entity_tag", 
                                                                         "given_entity_description", 
                                                                         "input_str"))
    # print(dataset)


    # define the teleprompter/optimizer
    teleprompter = LabeledFewShot(k=4)

    # compile them with labeled fewshot and training sets
    compiled_annotator_agent = teleprompter.compile(
        student=annotator_agent,
        trainset=dataset,
        )

    compiled_annotator_agent = assert_transform_module(compiled_annotator_agent, backtrack_handler)

   

    # dataset = pd.read_csv("./newsmediabias_plus_10rows.csv", nrows=10)


    for index, row in  tqdm(dataset.iterrows(), 
                            total=dataset.shape[0], 
                            desc="Evaluating Rows"):
        # try:
        input_str = row['article_text']
        print(type(input_str))
        annotation = annotate_ner_multi_label(input_str[:500])

        text_str = annotation['input_str']
        # print(len(input_str.split()))
        ner_tags = annotation['ner_tags']
        # print(len(ner_tags))
        rationale = annotation['rationale']
        ner_tags_str = str(ner_tags)

        # To print
        tokens = input_str.split()
        data = {
            "Token": tokens,
            "NER Tags": [' '.join(tags) for tags in ner_tags]
        }
        print(data)
        # df = pd.DataFrame(data)

        # print(df.head())
        # print(f"Annotated input str: {row['id']}, and inserted to db")

        # except Exception as e:
        #     print(f"Error adding to db: {str(e)}")