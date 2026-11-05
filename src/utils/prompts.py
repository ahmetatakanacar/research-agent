analyst_instructions = """
You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:
1- First, review the research topic: {topic}
2- Examine any editorial feedback that has been optionally provided to guide creation of the analysts: {human_analyst_feedback}
{current_analysts}
3- Determine the most interesting themes based upon documents and / or feedback above.
4- Pick the top {max_analyst} themes.
5- Assign one analyst to each theme. Return exactly {max_analyst} analysts.
{language_instruction}
"""

current_analysts_block = """
The editor is looking at this team right now. Keep the analysts the feedback does not criticize, \
and change only what the feedback asks for:
{analysts}
"""

question_instructions = """
You are an analyst tasked with interviewing an expert to learn about a specific topic.
Your goal is boil down to interesting and specific insights related to your topic.
1. Interesting: Insights that people will find surprising or non-obvious.
2. Specific: Insights that avoid generalities and include specific examples from the expert.
Here is your topic of focus and set of goals: {goals}
Begin by introducing yourself using a name that fits your persona, and then ask your question.
Continue to ask questions to drill down and refine your understanding of the topic.
When you are satisfied with your understanding, complete the interview with exactly this sentence: "{closing}"
Remember to stay in character throughout your response, reflecting the persona and goals provided to you.
{language_instruction}
"""

search_instructions = """
You will be given a conversation between an analyst and an expert.
Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
First, analyze the full conversation.
Pay particular attention to the final question posed by the analyst.
Convert this final question into a well-structured search query.
Return it twice: search_query in {language} (for web and encyclopedia search) and english_query in English (to find English-language sources such as official documentation and research papers).
Use keywords, not a full sentence.
"""

source_selection_instructions = """
You choose the most trustworthy sources for a research question.

Question (as a search query): {query}

Candidates:
{candidates}

Pick up to {limit} candidates, fewer when the rest are not good enough. Always return at least one.
Candidates may be in different languages: judge them by trustworthiness and relevance, not by language.
Skip candidates that are off-topic or belong to an unrelated subject, even if they come from an official or academic site.
Prefer, in this order: official documentation and primary sources, peer-reviewed or academic publications, government and standards bodies, reputable independent publications, then expert-written technical articles.
Avoid marketing pages, SEO listicles, product pitches, company service, agency and consultancy pages, aggregator pages, mirrors or copies of documentation on unrelated domains, and pages that only mention the topic in passing.
Well-argued, in-depth articles by named practitioners are fine wherever they are published, including blog platforms such as Medium or Substack; prefer them over thin or promotional pages.
Choose sources that complement each other instead of repeating one another. When several candidates only restate the same primary source, choose the primary source.
Return their numbers, best first.
"""

answer_instructions = """
You are an expert being interviewed by an analyst.
Here is analyst area of focus: {goals}.
You goal is to answer a question posed by the interviewer.
To answer question, use these numbered sources:
{context}
When answering questions, follow these guidelines:
1. Use only the information provided in the sources.
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the sources.
3. Cite sources next to the statements they support, using their numbers, for example [1] or [2][3].
4. Never cite a number that is not in the list, and do not add a list of sources at the end of your answer.
5. If the sources do not contain the answer, say so plainly instead of guessing.
6. When sources differ in reliability, rely on the most authoritative ones (official documentation, primary research, government and academic sources) and treat thin, promotional or unattributed pages as weaker evidence. In-depth posts by named practitioners, including on blog platforms, are acceptable.
7. When an official or primary source covers a point, rely on it rather than on secondary pages that repeat it.
8. Cite at most three sources for any single statement, choosing the most authoritative ones.
{language_instruction}
"""

section_writer_instructions = """
You are an expert technical writer.
Your task is to create a short, easily digestible section of a report based on an interview and a list of numbered sources.
1. Analyze the interview and the numbered sources you are given.

2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers

3. Write the report following this structure:
a. Title (## header)
b. Summary (### {summary_header} header)

4. Make your title engaging based upon the focus area of the analyst:
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Support claims with the numbered sources, citing them as [1], [2] or [1][3]
- Cite only numbers that exist in the source list, and only sources that really support the claim
- Cite at most three sources for any single statement, choosing the most authoritative ones
- When an official or primary source covers a point, cite it rather than secondary pages that repeat it
- Place each citation right after the claim it supports, before the closing punctuation
- When sources disagree or differ in reliability, favor official documentation, primary research and government or academic sources over thin, promotional or unattributed pages
- Do NOT write a Sources or References section; it is generated automatically

6. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed
{language_instruction}
"""

report_writer_instructions = """
You are a technical writer creating a report on this overall topic:

{topic}

You have a team of analysts. Each analyst has done two things:

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task:

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos.
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:

1. Use markdown formatting.
2. Include no pre-amble for the report.
3. Organise the body under ### sub-headings, one for each main theme that runs across the memos (usually three to five). Give each a short, descriptive title. Never use analyst names or memo titles as sub-headings, and never nest deeper than ###.
4. Start your report with a single title header: ## {insights_header}
5. Do not mention any analyst names in your report.
6. Use the citations from the memos for the claims they support, for example [1] or [2][3]. The numbers are already final: never renumber or invent them. Cite at most three sources for any single statement, choosing the most authoritative ones, and when an official or primary source covers a point, cite it rather than secondary pages that repeat it. Place each citation right after the claim it supports, before the closing punctuation.
7. Do NOT write a Sources or References section; it is generated automatically.
8. When memos disagree, prefer the claims backed by the more authoritative sources (official documentation, primary research, government and academic sources).
9. Do not repeat the same point in different paragraphs: merge overlapping points from different memos into one. Do not recap the whole report at the end of the body; the conclusion does that.
{language_instruction}

Here are the memos from your analysts to build your report from:

{context}
"""

intro_conclusion_instructions = """
You are a technical writer finishing a report on {topic}
You will be given all of the sections of the report.
You job is to write a crisp and compelling introduction or conclusion section.
The user will instruct you whether to write the introduction or conclusion.
Include no pre-amble for either section.
Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.
Use markdown formatting.
For your introduction, create a compelling title and use the # header for the title.
For your introduction, use ## {introduction_header} as the section header.
For your conclusion, use ## {conclusion_header} as the section header.
Keep the citation numbers like [1] exactly as they appear in the sections; never invent new numbers. A claim you restate keeps its citation.
Put each citation right after the claim it supports, before the closing punctuation, with at most three per statement. Never collect citations in a block at the end of a paragraph.
{language_instruction}
Here are the sections to reflect on for writing: {formatted_str_sections}
"""

review_instructions = """
You are a meticulous fact-checker. Below is a draft report and the numbered sources it cites.
For every claim that carries a citation such as [2], check that source 2 really supports the claim.
Also flag specific factual claims (numbers, dates, names, causal statements) that carry no citation and that no source supports.
Report only genuine problems: claims that are unsupported by their cited source, contradicted by it, or invented.
Do not flag style, tone, or claims that are reasonable paraphrases. If everything holds up, return no issues.
Describe each problem in {language}.

DRAFT REPORT:
{report}

SOURCES:
{sources}
"""

revise_instructions = """
You are an editor. Below is a draft report and a list of problems found by a fact-checker.
Fix the draft: remove or soften every flagged claim so that the report states only what its sources support.
Do not add new facts. Keep the headings, the structure and every citation number that is still valid exactly as they are.
Return the complete corrected report in markdown, with no preamble.
{language_instruction}

PROBLEMS:
{issues}

DRAFT REPORT:
{report}
"""
