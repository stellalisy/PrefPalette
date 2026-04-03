dimension_pairs = {
    "verbosity": "concise-verbose",
    "formality": "casual-formal", 
    "supportiveness": "toxic-supportive", 
    "sarcasm": "genuine-sarcastic", 
    "humor": "serious-humorous", 
    "politeness": "rude-polite",
    "assertiveness": "passive-assertive",
    "empathy": "detached-empathetic",
    "directness": "indirect-direct"
}

norms = ["verbosity", "formality", "supportiveness", "sarcasm", "humor", "politeness", "assertiveness", "empathy", "directness"]

norm_levels = {
    "verbosity": ["extremely concise", "somewhat concise", "neutral (concise-verbose)", "somewhat verbose", "extremely verbose"],
    "formality": ["extremely casual", "somewhat casual", "neutral (casual-formal)", "somewhat formal", "extremely formal"],
    "supportiveness": ["extremely toxic", "somewhat toxic", "neutral (toxic-supportive)", "somewhat supportive", "extremely supportive"],
    "sarcasm": ["extremely genuine", "somewhat genuine", "neutral (genuine-sarcastic)", "somewhat sarcastic", "extremely sarcastic"],
    "humor": ["extremely serious", "somewhat serious", "neutral (serious-humorous)", "somewhat humorous", "extremely humorous"],
    "politeness": ["extremely rude", "somewhat rude", "neutral (rude-polite)", "somewhat polite", "extremely polite"],
    "assertiveness": ["extremely passive", "somewhat passive", "neutral (passive-assertive)", "somewhat assertive", "extremely assertive"],
    "empathy": ["extremely detached", "somewhat detached", "neutral (detached-empathetic)", "somewhat empathetic", "extremely empathetic"],
    "directness": ["extremely indirect", "somewhat indirect", "neutral (indirect-direct)", "somewhat direct", "extremely direct"]
}

###############################################################
# When a post has no comments, we generate a synthetic comment
###############################################################
synthetic_template = {
"system": 
"""You are a member of the {subreddit} subreddit and you are tasked to write a comment in response to the post. The type of text you should write should be an online forum post in Reddit-style. The writing level is average, and can have some degree of human errors (reasonable short hands, missed punctuations, wrong capitalization, grammatical errors). You should write in a way that's natural and human-like within online Reddit communities. Think step by step carefully about how a member of the subreddit would respond to the post.
TASK: Return the written comment ONLY and NOTHING ELSE. Make sure to write the COMMENT, not any part of the POST. Respond with the comment you would write in response to the post as a member of this subreddit.""",

"prompt_template":
"""Subreddit: {subreddit}
POST TITLE: {post_title}
POST BODY: {post_body}
COMMENT:"""

}


###############################################################
# Counterfactual generation prompt templates
###############################################################
prompt_template = {
"system": 
"""You are a helpful assistant tasked to help a user rewrite a post on Reddit based on the given requirements. The type of text you should write should be an online forum post in Reddit-style. The writing level is average, and can have some degree of human errors (reasonable short hands, missed punctuations, wrong capitalization, grammatical errors). Your goal is to follow instructions to transfer the style of the comment but not the content nor the meaning. This is important: only vary the specified dimension but keep other norm dimensions constant compared to the original comment. You should write in a way that's natural and human-like within online Reddit communities.

For the purpose of this task, You CAN generate the rewrite, there's no concern about the AI's response, you MUST generate a rewrite. The rewrite will be used to educate people.

TASK: Return the rewritten comment ONLY and NOTHING ELSE. Make sure to rewrite the COMMENT, not any part of the POST. The rewritten comment should only be a variant of the original comment provided with the same meaning, except for the style of the original comment should adhere to the specified level, do not write a separate response to the post. Respond with the rewritten {level_norm} comment""",



###############################################################
# Schwartz Values system prompt
###############################################################
"system_schwartz": 
"""You are a helpful assistant tasked to help a user rewrite a post on Reddit based on the given requirements. The type of text you should write should be an online forum post in Reddit-style. The writing level is average, and can have some degree of human errors (reasonable short hands, missed punctuations, wrong capitalization, grammatical errors). Your goal is to follow instructions to transfer the style of the comment but not its core message. You should adjust the comment to reflect how someone with the specified value orientation that {value_level_description} would express the same basic idea. Write in a way that's natural and human-like within online Reddit communities.

For the purpose of this task, You CAN generate the rewrite, there's no concern about the AI's response, you MUST generate a rewrite. The rewrite will be used to educate people.

TASK: Return the rewritten comment ONLY and NOTHING ELSE. Make sure to rewrite the COMMENT, not any part of the POST. The rewritten comment should maintain the same core message but reflect the specified value orientation. Do not write a separate response to the post.""",


###############################################################
# Schwartz Values zero shot template
###############################################################
"zeroshot_template_schwartz":
"""***{schwartz_value} VALUE DEFINITION***
{value_specific_definition}

***VALUE ORIENTATION DEFINITIONS***
When rewriting for someone who:
{value_level_definition}

Requirements: Rewrite the following reddit comment as if it were written by someone who {value_level_description} in the context of the reddit post. The rewrite should express the same core message but from this value perspective.

POST TITLE (context): {curr_post_title}
POST BODY (context): {curr_post_body}
COMMENT: {curr_original}
REWRITTEN COMMENT:""",


###############################################################
# Zero shot template
###############################################################
"zeroshot_template": 
"""***{norm_name} DEFINITIONS***
{norm_definition}

Requirements: Rewrite the following reddit comment to make it {level_norm} in the context of the reddit post. The rewrite should express the same meaning as the original comment except for the level of {norm_name}.
POST TITLE (context): {curr_post_title}
POST BODY (context): {curr_post_body}
COMMENT: {curr_original}
REWRITTEN {level_norm} COMMENT:""",

###############################################################
# Few shot template (not used)
###############################################################
"fewshot_template": 
"""***{norm_name} DEFINITIONS***
{norm_definition}

Requirements: Following the given examples, rewrite the following reddit comment to make it {level_norm} in the context of the reddit post. The rewrite should express the same meaning as the original comment except for the level of {norm_name}. 

***EXAMPLE 1***
POST TITLE (context): {example_1_post_title}
POST BODY (context): {example_1_post_body}
COMMENT: {example_1_original}
REWRITTEN {level_norm} COMMENT: {example_1_rewritten}

***EXAMPLE 2***
POST TITLE (context): {example_2_post_title}
POST BODY (context): {example_2_post_body}
COMMENT: {example_2_original}
REWRITTEN {level_norm} COMMENT: {example_2_rewritten}

***EXAMPLE 3***
POST TITLE (context): {example_3_post_title}
POST BODY (context): {example_3_post_body}
COMMENT: {example_3_original}
REWRITTEN {level_norm} COMMENT: {example_3_rewritten}

Now you do it:
POST TITLE (context): {curr_post_title}
POST BODY (context): {curr_post_body}
COMMENT: {curr_original}
REWRITTEN {level_norm} COMMENT:""",

###############################################################
# Multi-turn few shot template (not used)
###############################################################
"multiturn_fewshot_template": 
"""***{norm_name} DEFINITIONS***
{norm_definition}

Requirements: Rewrite the following reddit comment to make it {level_norm} in the context of the reddit post. The rewrite should express the same meaning as the original comment except for the level of {norm_name}.""",

"multiturn_intermediate_template":
"""POST TITLE (context): {example_post_title}
POST BODY (context): {example_post_body}
COMMENT: {example_original}
REWRITTEN {level_norm} COMMENT:"""
}




###############################################################
# Verifier prompt template
###############################################################
verifier_template = {
"system":
"""You are a linguistic expert tasked with comparing which linguistic dimension is more present between two Reddit comments. Please act as an impartial judge and evaluate the {norm_name} of two comments to a reddit post displayed below. You should choose the comment that is {less_more_norm}. Your evaluation should follow the guidelines provided below on the likert scale definition of {norm_name}.""",

"zeroshot_template": 
"""***{norm_name} DEFINITION***
{dimension_definition}

Only use the provided post title and post description as context. Begin your evaluation by comparing the two comments on their {norm_name} and provide a short one-sentence rationale. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names in the comment. Be as objective as possible. Do not allow the decision to be influenced by the order by which the comments are presented to you, your answer should be consistent if the order of the comments are switched. After providing your explanation, output your final verdict by strictly following this format: "RATIONALE: one-sentence rationale. [[A]]" if comment A is better, "RATIONALE: one-sentence rationale. [[B]]" if comment B is better.

POST TITLE (context): {curr_post_title}
POST BODY (context): {curr_post_body}
COMMENT A: {comment_a}
COMMENT B: {comment_b}
RATIONALE:"""
}


###############################################################
# Norm definitions
###############################################################
norm_definitions = {
"verbosity":
"""Extremely Concise: Very brief or telegraphic statements (e.g., a few words or short sentences). Leaves out nonessential details or explanations.
Somewhat Concise: Short sentences, generally sticking to key points with minimal elaboration. Adds brief clarifications if needed but avoids long descriptions.
Neutral: Uses a moderate amount of detail; not too wordy or too sparse. Balances essential information with some explanatory content.
Somewhat Verbose: Includes additional information, sometimes more than strictly necessary. May repeat points or use multiple sentences to clarify each idea.
Extremely Verbose: Provides lengthy explanations or tangential details. Often repeats ideas or includes extra commentary that goes beyond the main point.""",

"formality":
"""Extremely Casual: Very informal style, might include slang (e.g., "y'all," "gonna," "lol") or text/chat abbreviations. Loose grammar, relaxed punctuation; feels like casual chat.
Somewhat Casual: Generally informal but with fewer slang terms. Comfortable, conversational tone without strict adherence to formalities.
Neutral: Standard language use, avoids heavy slang or overly formal phrases. Straightforward diction without a pronounced casual or formal tilt.
Somewhat Formal: Polished language, generally avoids slang. More structured sentences and courteous phrasing.
Extremely Formal: Highly professional or academic tone (e.g., "I would like to request…," "It is imperative…"). Strict grammar, no slang, uses titles or polite forms of address if relevant.""",

"supportiveness":
"""Extremely Toxic: Very hostile or insulting language, includes slurs or personal attacks. Conveys strong contempt, aggression, or hatred.
Somewhat Toxic: Negative or harsh tone, may use mild insults or belittling remarks. Often dismissive or critical without offering constructive points.
Neutral: Neither supportive nor hostile. Mostly factual or matter-of-fact statements, no encouragement or aggression.
Somewhat Supportive: Generally positive or kind tone, shows some empathy or encouragement. May include mild praise, reassurance, or understanding.
Extremely Supportive: Warm, uplifting, actively encouraging, and compassionate. Shows clear emotional support or enthusiasm for others.""",

"sarcasm":
"""Extremely Genuine: Completely sincere, transparent intentions. Language is direct, earnest, and free from any hint of mockery.
Somewhat Genuine: Mostly honest and straightforward, though might include occasional playful or ironic remarks. Overall intent is sincere, with only mild hints of sarcasm if any.
Neutral: Does not clearly convey earnest warmth or sarcastic edge. Straight, factual statements without emotional or ironic overtones.
Somewhat Sarcastic: Noticeable irony, mocking, or playful jabs in some parts. Tone suggests the speaker is poking fun or being indirectly critical.
Extremely Sarcastic: Heavy use of biting wit, irony, or mocking language throughout. May appear intentionally cutting or cynically humorous.""",

"humor":
"""Extremely Serious: No jokes or lighthearted remarks; tone is grave or earnest. Focuses strictly on factual or urgent matters.
Somewhat Serious: Primarily serious with minimal humor (e.g., a small side comment). Overall, keeps a reserved, sober tone.
Neutral: Neither strongly serious nor comedic. Balanced approach—may include mild hints of humor or seriousness.
Somewhat Humorous: Injects occasional wit, jokes, or playful language. Keeps the overall message but adds a clear lighthearted tone.
Extremely Humorous: Consistently comedic or playful; heavily reliant on jokes, puns, or funny stories. Often exaggerates or entertains at length for comedic effect.""",

"politeness":
"""Extremely Rude: Highly offensive or disrespectful language; direct insults, name-calling. Dismissive or contemptuous tone with no courtesy.
Somewhat Rude: Shows disrespect or blunt negativity in parts. May be curt, insensitive, or use mildly offensive language.
Neutral: Neither particularly polite nor rude, simply direct. No evident courtesy but also avoids insults.
Somewhat Polite: Generally respectful language, might use polite expressions ("please," "thank you"). A considerate tone with occasional small gestures of courtesy.
Extremely Polite: Very courteous, uses consistently respectful language. Politely phrases even critical opinions, consistently mindful of others' feelings.""",

"assertiveness":
"""Extremely Passive: Highly deferential, overly apologetic, frequently uses hedging language (e.g., "maybe," "just," "sort of"). Avoids stating opinions directly, phrases thoughts as questions, and readily yields to contradicting views.
Somewhat Passive: Moderately tentative, occasionally uses hedging phrases but will express opinions. Tends to soften statements with qualifiers and often acknowledges alternative viewpoints.
Neutral: Balances confidence with consideration. States opinions clearly without being overly forceful or unnecessarily hesitant. Uses a mix of direct statements and open-ended language.
Somewhat Assertive: Confidently presents views with minimal hedging. Uses declarative statements, makes clear recommendations, and doesn't downplay expertise or knowledge.
Extremely Assertive: Highly confident, authoritative tone. Makes definitive statements, uses commanding language (e.g., "you need to," "absolutely"), and presents opinions as facts. Shows unwavering conviction without qualifiers.
When rewriting, diversify the ways in which to express passivity or assertiveness. For example, don't use basic expressions like "If I'm not mistaken" or "I'm guessing", you can still use them, but just not too often.""",

"empathy":
"""Extremely Detached: Purely logical/factual with no acknowledgment of emotions or personal circumstances. Focuses exclusively on objective analysis without considering how others might feel.
Somewhat Detached: Primarily focused on facts and logic but occasionally acknowledges emotional aspects. May recognize feelings but doesn't deeply engage with them.
Neutral: Balances logical analysis with basic emotional awareness. Acknowledges feelings when relevant without making them the central focus.
Somewhat Empathetic: Actively considers emotional perspectives alongside factual elements. Shows understanding of others' situations and validates feelings. Uses supportive language.
Extremely Empathetic: Deeply attuned to emotional aspects, prioritizes understanding and validating feelings. Uses compassionate language, explicitly acknowledges pain points, offers emotional support, and demonstrates genuine concern for others' wellbeing.""",

"directness":
"""Extremely Indirect: Heavily relies on euphemisms, hints, and implicit communication. Avoids stating points clearly, uses excessive qualifiers, and buries main points within tangential information. May talk around sensitive topics rather than addressing them.
Somewhat Indirect: Takes time to build up to main points, uses some hedging language and softening phrases. Tends to be diplomatic and may imply rather than explicitly state criticisms.
Neutral: Balances clarity with tact. Makes points relatively clearly but with appropriate contextual framing and consideration of reception.
Somewhat Direct: Gets to the point quickly with minimal preamble. States opinions and feedback clearly with limited hedging or softening language.
Extremely Direct: Bluntly states opinions and thoughts with no sugar-coating. Uses concise, explicit language without hedging or diplomatic phrasing. Addresses sensitive topics head-on without evasion.
When rewriting, diversify the ways in which to express directness or indirectness. For example, try to avoid basic expressions like "If I'm not mistaken", you can still use them, but just not too often.""",
}



schwartz_values = ["Self-Direction", "Stimulation", "Hedonism", "Achievement", "Power", "Security", "Conformity", "Tradition", "Benevolence", "Universalism"]

schwartz_value_definitions = {
    "Self-Direction": "Self-Direction centers on independent thought and action—choosing, creating, exploring, and making autonomous decisions.",
    
    "Stimulation": "Stimulation centers on excitement, novelty, challenge, and variety in life.",
    
    "Hedonism": "Hedonism centers on pleasure, enjoyment, and sensuous gratification for oneself.",
    
    "Achievement": "Achievement centers on personal success through demonstrating competence according to social standards, ambition, and recognition for accomplishments.",
    
    "Power": "Power centers on social status, prestige, control or dominance over people and resources, authority, and wealth.",
    
    "Security": "Security centers on safety, harmony, stability of society, relationships, and self; avoiding threats and maintaining order.",
    
    "Conformity": "Conformity centers on restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms; following rules and self-discipline.",
    
    "Tradition": "Tradition centers on respect, commitment, and acceptance of the customs and ideas that one's culture, religion, or family provides; humility and devotion to established practices.",
    
    "Benevolence": "Benevolence centers on preserving and enhancing the welfare of those with whom one is in frequent personal contact; helpfulness, honesty, forgiveness, and loyalty toward friends and family.",
    
    "Universalism": "Universalism centers on understanding, appreciation, tolerance, and protection for the welfare of all people and nature; social justice, equality, and environmental preservation."
}

value_level_definitions = {
    "Self-Direction": """
"Strongly Against Self-Direction": Emphasize reliance on established rules, traditions, or authorities. Express discomfort with independence and preference for clear guidelines from others. Include phrases like "we should follow what experts/most people say" or "I prefer to stick with proven methods." Show skepticism toward unconventional thinking.
"Somewhat Against Self-Direction": Show moderate deference to conventional wisdom while occasionally suggesting personal opinions. Include qualifiers like "I generally follow what's recommended, but..." Prioritize established approaches over novel ones.
"Neutral Toward Self-Direction": Balance personal views with consideration of existing norms. Express both individual perspectives and acknowledgment of established practices without strongly favoring either.
"Somewhat Values Self-Direction": Emphasize personal judgment and independent reasoning. Include phrases suggesting personal research or thinking ("I've thought about this and decided..."). Show comfort questioning conventional approaches when they don't make sense personally.
"Strongly Values Self-Direction": Emphasize complete autonomy in thinking and decision-making. Include strong statements about personal choice ("I make my own decisions regardless of what's conventional"). Express resistance to unnecessary constraints and enthusiasm for creative, novel approaches. Challenge authority when it limits personal freedom.""",

    "Stimulation": """
"Strongly Against Stimulation": Emphasize comfort with routine, predictability, and familiar experiences. Express discomfort or anxiety about new, untested, or exciting situations. Include phrases like "I prefer to stick with what's familiar" or "why risk trying something new?"
"Somewhat Against Stimulation": Show preference for mostly familiar experiences with occasional, carefully controlled novelty. Express caution about major changes or highly stimulating activities.
"Neutral Toward Stimulation": Balance interest in some new experiences with appreciation for familiar routines. Neither actively seek nor avoid excitement and change.
"Somewhat Values Stimulation": Express enthusiasm for new experiences and variety. Include phrases suggesting boredom with routine ("I needed something different"). Show willingness to try novel approaches and excitement about potential discoveries.
"Strongly Values Stimulation": Emphasize constant pursuit of excitement, novelty, and challenges. Include phrases expressing boredom with the status quo and enthusiasm for risk-taking. Express active avoidance of routine and predictability ("I can't stand doing the same thing every day"). Show eagerness to embrace the unknown and unconventional.""",

    "Hedonism": """
"Strongly Against Hedonism": Emphasize self-restraint, delayed gratification, and skepticism toward pursuit of pleasure. Express concern about indulgence and preference for practicality, duty, or spiritual values over enjoyment. Include phrases like "we shouldn't just do what feels good" or references to the virtue of self-denial.
"Somewhat Against Hedonism": Show moderate restraint and prioritization of responsibilities over pleasure. Express some wariness about excessive enjoyment while accepting small pleasures in moderation.
"Neutral Toward Hedonism": Balance enjoyment with other priorities. Acknowledge the value of pleasure without making it primary. Express neither strong self-denial nor enthusiastic pursuit of gratification.
"Somewhat Values Hedonism": Emphasize the importance of enjoying life and finding pleasure in experiences. Include phrases about "treating yourself" or "making the most of life." Show willingness to prioritize enjoyable activities alongside responsibilities.
"Strongly Values Hedonism": Emphasize maximizing pleasure and enjoyment as central life priorities. Include phrases about "living in the moment" and "life is too short not to enjoy." Express skepticism toward unnecessary self-restraint and enthusiasm for indulgence when desired. Make references to sensory experiences and personal gratification.""",

    "Achievement": """
"Strongly Against Achievement": Emphasize skepticism toward competitive success, status-seeking, and conventional accomplishment. Express preference for cooperation over competition and contentment over ambition. Include phrases questioning the value of traditional markers of success ("who cares about promotions/awards/status?").
"Somewhat Against Achievement": Show limited interest in competition or status while focusing more on personal satisfaction or cooperation. Express moderate ambition but question intense pursuit of achievement markers.
"Neutral Toward Achievement": Balance interest in some personal accomplishments with other life priorities. Neither strongly pursue nor reject conventional success markers. Express moderate satisfaction from achievement without making it central.
"Somewhat Values Achievement": Emphasize the importance of success, competence, and recognition. Include references to goals, personal excellence, and competitive improvement. Show pleasure in accomplishments and some focus on demonstrating capability to others.
"Strongly Values Achievement": Emphasize intense ambition, competitive excellence, and recognition as central motivations. Include phrases about being "the best" or "reaching the top." Express strong drive to demonstrate competence, receive acknowledgment for accomplishments, and attain measurable success. Reference social standards of achievement and personal status directly.""",

    "Power": """
"Strongly Against Power": Emphasize discomfort with hierarchies, control, and status differences. Express preference for egalitarianism and disinterest in authority positions. Include phrases questioning power structures ("why should anyone have authority over others?") and expressions of discomfort with controlling resources or people.
"Somewhat Against Power": Show limited interest in status or authority while accepting their necessity in some contexts. Express preference for collaboration over control and skepticism toward status-seeking behaviors.
"Neutral Toward Power": Balance recognition of power structures with neither strong desire nor aversion to authority positions. Express moderate interest in influence without making dominance or status primary concerns.
"Somewhat Values Power": Emphasize the importance of having influence, control, and respected status. Include references to decision-making authority and resources as desirable. Show comfort directing others and having recognized social standing.
"Strongly Values Power": Emphasize desire for dominance, high status, control over situations, and influence over others. Include phrases about "being in charge" or "having the final say." Express strong concern with social ranking, wealth as power, and authority as essential. Reference competitions for dominance or status directly.""",

    "Security": """
"Strongly Against Security": Emphasize comfort with uncertainty, change, and potential risk. Express skepticism toward excessive caution and safety measures. Include phrases questioning the value of stability ("change keeps things interesting") and resistance to security-based restrictions.
"Somewhat Against Security": Show moderate comfort with some uncertainty and change. Express willingness to accept reasonable risks and question excessive focus on safety and stability.
"Neutral Toward Security": Balance concern for basic safety with acceptance of some uncertainty. Neither strongly pursue security measures nor reject them as unnecessary. Express moderate caution without making it a primary focus.
"Somewhat Values Security": Emphasize the importance of safety, stability, and predictability. Include references to potential risks and the need for caution. Show preference for well-established, secure options over uncertain alternatives.
"Strongly Values Security": Emphasize intense concern with threats, dangers, and instability. Include phrases about "staying safe" or "avoiding unnecessary risks." Express strong desire for order, consistency, and protection from potential harm. Reference societal stability, personal security, and health safety as central concerns.""",

    "Conformity": """
"Strongly Against Conformity": Emphasize rejection of arbitrary social norms and unnecessary rules. Express comfort breaking conventions and questioning established expectations. Include phrases like "why follow rules that don't make sense?" or "I don't care what people think is 'proper'."
"Somewhat Against Conformity": Show moderate skepticism toward some social expectations while accepting others. Express willingness to question certain norms while acknowledging the value of basic social cohesion.
"Neutral Toward Conformity": Balance respect for important social norms with flexibility about less consequential ones. Neither strongly rebel against nor strictly adhere to all conventions. Express situational judgment about when to conform.
"Somewhat Values Conformity": Emphasize the importance of appropriate behavior, politeness, and following established rules. Include references to social expectations and the value of self-discipline. Show discomfort with actions that might disrupt social harmony.
"Strongly Values Conformity": Emphasize strict adherence to social norms, proper conduct, and established rules. Include phrases about "what's expected" or "how things should be done." Express strong disapproval of behavior that violates conventions and emphasis on self-restraint. Reference respect for authority figures and established guidelines as essential.""",

    "Tradition": """
"Strongly Against Tradition": Emphasize skepticism toward inherited customs, religious practices, or family traditions. Express preference for contemporary, progressive approaches over established ones. Include phrases questioning the value of tradition ("just because it's always been done that way doesn't make it right").
"Somewhat Against Tradition": Show limited appreciation for some meaningful traditions while questioning others. Express preference for adapting or updating traditional practices rather than preserving them unchanged.
"Neutral Toward Tradition": Balance respect for meaningful traditions with openness to new approaches. Neither strongly preserve traditions for their own sake nor dismiss them as irrelevant. Express selective engagement with customs based on their current value.
"Somewhat Values Tradition": Emphasize respect for established customs, religious or cultural practices, and inherited wisdom. Include references to the importance of honoring traditions and learning from the past. Show preference for time-tested approaches over novel ones.
"Strongly Values Tradition": Emphasize deep reverence for established customs, religious doctrines, and cultural heritage. Include phrases about "honoring our ancestors" or "respecting time-tested ways." Express strong commitment to preserving and practicing traditions unchanged and skepticism toward modernizing influences. Reference humility before established practices and devotion to cultural or religious identities.""",

    "Benevolence": """
"Strongly Against Benevolence": Emphasize focus on self-interest over helping close others. Express skepticism toward sacrificing personal interests for friends or family. Include phrases questioning obligations to others ("why should I go out of my way?") and preference for transactional relationships.
"Somewhat Against Benevolence": Show limited concern for others' welfare while maintaining focus on personal interests. Express selective helpfulness when convenient and wariness about excessive self-sacrifice.
"Neutral Toward Benevolence": Balance care for close others with self-interest. Neither consistently prioritize helping others nor focus exclusively on personal concerns. Express situation-dependent willingness to assist friends and family.
"Somewhat Values Benevolence": Emphasize care and concern for the welfare of friends and family. Include references to helping close others and maintaining positive relationships. Show willingness to make reasonable sacrifices for those in one's immediate circle.
"Strongly Values Benevolence": Emphasize intense concern for the happiness and wellbeing of close others as a primary life focus. Include phrases about "being there for people who matter" or "putting loved ones first." Express strong willingness to sacrifice personal interests for friends and family and deep commitment to honesty, forgiveness, and loyalty in relationships. Reference helping behavior and emotional support as essential priorities.""",

    "Universalism": """
"Strongly Against Universalism": Emphasize skepticism toward broad concerns about all humanity or nature. Express preference for focusing on one's own group rather than universal concerns. Include phrases questioning global equality efforts ("we should take care of our own first") and environmental protection when it conflicts with human interests.
"Somewhat Against Universalism": Show limited concern for broad social issues while focusing more on immediate circles. Express selective interest in certain universal causes while questioning others, particularly when they require personal sacrifice.
"Neutral Toward Universalism": Balance concern for some broader human and environmental issues with other priorities. Neither strongly advocate for nor dismiss universal concerns. Express moderate, situational interest in equality and protection for all.
"Somewhat Values Universalism": Emphasize concern for fairness, equality, and environmental protection beyond one's immediate group. Include references to understanding diverse perspectives and caring about global issues. Show willingness to consider the welfare of all people and nature in decisions.
"Strongly Values Universalism": Emphasize deep commitment to equality, social justice, and environmental protection as central life priorities. Include phrases about "equal rights for everyone" or "protecting the planet." Express strong concern for understanding different perspectives, tolerance for diversity, and protection of vulnerable groups and natural systems. Reference global welfare and sustainability directly as essential considerations."""
}

value_level_descriptions = {
    1: lambda value: f"strongly opposes {value}",
    2: lambda value: f"is somewhat against {value}",
    3: lambda value: f"is neutral toward {value}",
    4: lambda value: f"somewhat values {value}",
    5: lambda value: f"strongly values {value}"
}

# Example usage:
def create_prompt_schwartz(schwartz_value, value_level, curr_post_title, curr_post_body, curr_original):
    value_specific_definition = schwartz_value_definitions[schwartz_value]
    value_level_definitions_for_this_value = value_level_definitions[schwartz_value]
    value_level_description = value_level_descriptions[value_level](schwartz_value)
    
    # Construct the prompt using these values
    prompt = prompt_template["zeroshot_template_schwartz"].format(
        schwartz_value=schwartz_value,
        value_specific_definition=value_specific_definition,
        value_level_definition=value_level_definitions_for_this_value,
        value_level_description=value_level_description,
        curr_post_title=curr_post_title,
        curr_post_body=curr_post_body,
        curr_original=curr_original
    )

    messages = [{"role": "system", "content": prompt_template["system_schwartz"].format(value_level_description=value_level_description)},
                {"role": "user", "content": prompt}]
    return messages
    
    

def get_multiturn_fewshot_prompts(norm_name, level, example_titles, example_posts, example_comments, example_rewrites, curr_post_title, curr_post_body, curr_original):
    level_norm = norm_levels[norm_name][level-1]
    system = prompt_template["system"].format(level_norm=level_norm)
    zeroshot_template = prompt_template["zeroshot_template"].format(norm_definition=norm_definitions[norm_name], level_norm=level_norm, norm_name=norm_name, curr_post_title=example_posts[0], curr_post_body=example_posts[0], curr_original=example_comments[0])
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": zeroshot_template}]

    for i in range(1, len(example_titles)):
        messages.append({"role": "assistant", "content": example_rewrites[i-1]})
        messages.append({"role": "user", "content": prompt_template["multiturn_intermediate_template"].format(example_post_title=example_titles[i], example_post_body=example_posts[i], example_original=example_comments[i], level_norm=level_norm)})
    
    messages.append({"role": "assistant", "content": example_rewrites[-1]})
    messages.append({"role": "user", "content": prompt_template["multiturn_intermediate_template"].format(example_post_title=curr_post_title, example_post_body=curr_post_body, example_original=curr_original, level_norm=level_norm)})

    return messages

def get_fewshot_prompts(norm_name, level, example_titles, example_posts, example_comments, example_rewrites, curr_post_title, curr_post_body, curr_original):
    level_norm = norm_levels[norm_name][level-1]
    system = prompt_template["system"].format(level_norm=level_norm)
    prompt = prompt_template["fewshot_template"].format(
        norm_definition=norm_definitions[norm_name], level_norm=level_norm, norm_name=norm_name, 
        example_1_post_title=example_titles[0], example_1_post_body=example_posts[0], example_1_original=example_comments[0], example_1_rewritten=example_rewrites[0], 
        example_2_post_title=example_titles[1], example_2_post_body=example_posts[1], example_2_original=example_comments[1], example_2_rewritten=example_rewrites[1], 
        example_3_post_title=example_titles[2], example_3_post_body=example_posts[2], example_3_original=example_comments[2], example_3_rewritten=example_rewrites[2], 
        curr_post_title=curr_post_title, curr_post_body=curr_post_body, curr_original=curr_original)
    
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    return messages

def get_zeroshot_prompts(norm_name, level, curr_post_title, curr_post_body, curr_original):
    level_norm = norm_levels[norm_name][level-1]
    system = prompt_template["system"].format(level_norm=level_norm)
    prompt = prompt_template["zeroshot_template"].format(
            norm_definition=norm_definitions[norm_name], level_norm=level_norm, norm_name=norm_name, 
            curr_post_title=curr_post_title, curr_post_body=curr_post_body, curr_original=curr_original)
        
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
    return messages

def get_verifier_prompt(norm_name, curr_post_title, curr_post_body, comment_a, comment_b):
    directions = [f"less {dimension_pairs[norm_name].split('-')[0]} / more {dimension_pairs[norm_name].split('-')[1]}", f"less {dimension_pairs[norm_name].split('-')[1]} / more {dimension_pairs[norm_name].split('-')[0]}"]
    directions = [f"less {dimension_pairs[norm_name].split('-')[0]}", # less rude == more polite
                  f"more {dimension_pairs[norm_name].split('-')[1]}", # more polite == less rude
                  f"less {dimension_pairs[norm_name].split('-')[1]}", # less polite == more rude
                  f"more {dimension_pairs[norm_name].split('-')[0]}"] # more rude == less polite
    
    system_prompt_11 = verifier_template["system"].format(norm_name=norm_name, less_more_norm=directions[0])
    system_prompt_12 = verifier_template["system"].format(norm_name=norm_name, less_more_norm=directions[1])
    system_prompt_21 = verifier_template["system"].format(norm_name=norm_name, less_more_norm=directions[2])
    system_prompt_22 = verifier_template["system"].format(norm_name=norm_name, less_more_norm=directions[3])
    
    zeroshot_prompt_a = verifier_template["zeroshot_template"].format(norm_name=norm_name, dimension_definition=norm_definitions[norm_name], curr_post_title=curr_post_title, curr_post_body=curr_post_body, comment_a=comment_a, comment_b=comment_b) 
    zeroshot_prompt_b = verifier_template["zeroshot_template"].format(norm_name=norm_name, dimension_definition=norm_definitions[norm_name], curr_post_title=curr_post_title, curr_post_body=curr_post_body, comment_a=comment_b, comment_b=comment_a) 

    messages = [[{"role": "system", "content": system_prompt_11},
                {"role": "user", "content": zeroshot_prompt_a}], # correct answer: B
                [{"role": "system", "content": system_prompt_12},
                {"role": "user", "content": zeroshot_prompt_a}], # correct answer: B
                [{"role": "system", "content": system_prompt_11},
                {"role": "user", "content": zeroshot_prompt_b}], # correct answer: A
                [{"role": "system", "content": system_prompt_12},
                {"role": "user", "content": zeroshot_prompt_b}], # correct answer: A
                [{"role": "system", "content": system_prompt_21},
                {"role": "user", "content": zeroshot_prompt_a}], # correct answer: A
                [{"role": "system", "content": system_prompt_22},
                {"role": "user", "content": zeroshot_prompt_a}], # correct answer: A
                [{"role": "system", "content": system_prompt_21},
                {"role": "user", "content": zeroshot_prompt_b}], # correct answer: B
                [{"role": "system", "content": system_prompt_22},
                {"role": "user", "content": zeroshot_prompt_b}], # correct answer: B
        ] 
                
    return messages
