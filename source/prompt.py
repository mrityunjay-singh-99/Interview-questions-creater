prompt_template = """
you are the expert at creating questions based on coding matrials and documentation.
your goal is to prepare a coder or programmer for exam and coding tests.
yoo do this by asking questions about the text below:

-------
{text}
-------

creating questions that will prepare the coders or programmers for the their tests.
Make sure not to loss any important infromation.

Questions:
"""


refine_template = """
you are an expert at creating practice questions based on coding matrials and documents.
you goal is to help a coder or programmer prepare for a coding test.
we have received some practice questions to a certain extent: {existing_answer}.
we have options to refine the existing questionas or add new ones.
(only if necessary) with some more context below.
-------
{text}
-------

Give the new context, refine the original questiona in English.
if the context is not helpful, please provide the original questions.
QUESTIONS:
"""