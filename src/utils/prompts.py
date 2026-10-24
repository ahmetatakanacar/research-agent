analyst_instructions= """
You are tasked with creating a set of AI analyst presonas. Follow these instructions carefully:
1- First, review the research topic: {topic}
2- Examine any editorial feedback that has been optionally provided to guide creation of the analyst: {human_analyst_feedback}
3- Determine the most interesting themes based up on documents and / or feedback above.
4- Pick the top {max_analyst} themes.
5- Assign one analyst to each theme.

"""