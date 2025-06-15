import re

def prompt(demo_type=None):
        """
        Function to return the prompt for each demographic type.
        Args:
            demo_type (str): The type of demographic information to extract.
        Returns:
            str: The prompt for the specified demographic type.
        """
        if demo_type=="TARGETED_GROUP":
            inst="""
            You are an AI expert and capable of analyzing text for various use cases. You are given a news article and asked to answer the following question.
            Task:
            Read through the news article, and indentify the targeted groups. In your analysis consider the following details:
            --A targeted group is a specific segment of a population that is intentionally focused on for particular actions, strategies, or interventions based on shared characteristics or needs.
            --Indentify only the name of the group and do not return more than 5 groups.
            Return your answers as the following format.
            
            TARGETED_GROUP={
                [targeted group/targeted groups]
                }

            """
        
        if demo_type=="GENDER":
            inst="""
            You are an AI expert and capable of analyzing text for various use cases. You are given a news article and asked to answer the following question.
            Task:
            Read through the news article, and indentify the persons and their genders. In your analysis consider the following details:
            --Consider only "Male", "Female" and "LGBTQ".
            
            Return your answers as the following format.
            GENDER={
                [Person]: [gender]
            }

            """
        if demo_type=="RACE":
            inst="""
                You are an AI expert and capable of analyzing text for various use cases. You are given a news article and asked to answer the following question.
                Task:
                Read through the news article, and indentify the race. In your analysis consider the following details:
                --Emphasize only on the following races: "White/Caucasian"," Black or African American", "Asian", "Hispanic/Latino", "Native American", and "Pacific Islander".
                Return your answers as the following format.
                RACE={
                [race/races]
                }

                """
        if demo_type=="RELIGION":
            inst="""
            You are an AI expert and capable of analyzing text for various use cases. You are given a news article and asked to answer the following question.
            Task:
            Read through the news article, and indentify the religions mentioned in the article. In your analysis consider the following details:
            --Consider only the following religions:  "Christian", "Muslim", "Jewish", "Hindu", and "Buddhist".
            Return your answers as the following format.
            
            RELIGION={
                [religion/religions]
            }

            """ 
        if demo_type=="TOPICS":
            
            inst="""
            You are an AI expert and capable of analyzing text for various use cases. You are given a news article and asked to perform the following tasks.
            Task:
            Read through the news article, and indentify the topics dicussed in the article. In your analysis consider the following details:
            --If the there is more than one topics discussed.
            --Consider reporting maximum 3 topics.
            Return your answers as the following format.
            
            TOPICS={
                [topic/topics]
            }

            """
        if demo_type=="IDIOLOGY":
            inst="""
            You are an AI expert and capable of analyzing text for various use cases. You are given a news article and asked to perform the following tasks.
            Task:
            Read through the news article, and indentify political idiology expressed in the article. In your analysis consider the following details:
            --Left-wing: Left-wing political ideology emphasizes social equality, progressive reforms, and a greater role for government in addressing societal issues. 
            It advocates for policies like higher taxes on the wealthy, social welfare programs, and environmental protections. Historically rooted in the French Revolution, 
            left-wing politics ranges from center-left positions that accept regulated capitalism to far-left views that may reject capitalism entirely in favor of socialism
            
            --Right-wing: Right-wing political ideology emphasizes tradition, hierarchy, and free-market capitalism. It advocates for limited government intervention in the economy, lower taxes, and conservative social values. 
            Right-wing politics often prioritizes individual responsibility, strong national defense, and maintaining existing social structures, while opposing rapid social change and left-wing policies like wealth redistribution. 
            The ideology ranges from moderate conservatism to far-right nationalism, appealing to nationalism and traditional family values.
            
            --Central: Centrism is a political ideology that occupies the middle ground between left-wing and right-wing beliefs. It is characterized by moderate policies that balance elements from both sides, emphasizing pragmatism, compromise, and gradual change. 
            Centrists often support a welfare state with moderate redistributive policies and seek to accommodate diverse viewpoints, making them crucial in coalition governments.
            --If no idiology is identified return "No Political idiology found".
            
            Do do return long text, just provide the idiology. Return your answers as the following format.
            
            IDIOLOGY={
                [political idiology]
            }

            """
            
                
        return inst

def extract_info(document, demo_type=None):
    """
    Function to extract demographic information from a document based on the specified type.
    Args:
        document (str): The document from which to extract demographic information.
        demo_type (str): The type of demographic information to extract.
    Returns:
        str: The extracted demographic information in a specific format.
    """
    demographics = None

    #print(demo_type)
    if demo_type == "TARGETED_GROUP":
        match = re.search(r'TARGETED_GROUP\s*=\s*\{([^}]+)\}', document)
    elif demo_type == "RACE":
        match = re.search(r'RACE\s*=\s*\{([^}]+)\}', document)
    elif demo_type == "GENDER":
        match = re.search(r'GENDER\s*=\s*\{([^}]+)\}', document)
    elif demo_type == "RELIGION":
        match = re.search(r'RELIGION\s*=\s*\{([^}]+)\}', document)
    elif demo_type == "TOPICS":
        match = re.search(r'TOPICS\s*=\s*\{([^}]+)\}', document)
    elif demo_type == "IDIOLOGY":
        match = re.search(r'IDIOLOGY\s*=\s*\{([^}]+)\}', document)
    else:
        return None  # Invalid demo_type
    #print(match)
    if match:
        demographics = match.group(1).strip()

    return demographics

