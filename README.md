# Setup
pip install -r requirements.txt

# generate workflow
python workflow_generator.py \
	--model groq/openai/gpt-oss-120b \
	--user_file prompts/user_prompt/mendelian_genetics_known_plant.txt \
	--disable_sub \
	--generate_diagram \


# arguments
	--disable_rag \
	--disable_sub \
	--max_depth 1 \
	--generate_diagram \

When you set max_depth=0, subflow is not created.

## model
	--model groq/moonshotai/kimi-k2-instruct-0905 \
	--model openrouter/moonshotai/kimi-k2:free \
	--model groq/qwen/qwen3-32b \
	--model gemini/gemini-2.5-flash \
	--model gemini/gemini-2.5-flash-lite \
	--model openrouter/google/gemma-3-27b-it:free \
	--model groq/openai/gpt-oss-120b \
	--model openrouter/openai/gpt-oss-120b:free \
	--model groq/meta-llama/llama-4-maverick-17b-128e-instruct \
	--model openrouter/meta-llama/llama-3.3-70b-instruct:free \

## user_file
	--user_file prompts/user_prompt/boil.txt \
	--user_file prompts/user_prompt/chemistry_mix_paint_secondary_color.txt \
	--user_file prompts/user_prompt/grow_plant.txt \
	--user_file prompts/user_prompt/measure_melting_point.txt  \
	--user_file prompts/user_prompt/measure_melting_point-0.txt \
	--user_file prompts/user_prompt/measure_melting_point-melco.txt \
	--user_file prompts/user_prompt/mendelian_genetics_known_plant.txt \
	--user_file prompts/user_prompt/test_conductivity.txt \


# create diagram
python output/diagram.py \
output/measure_melting_point-melco/gpt-oss-120b-final-202512250152.txt

# register subflows to DB
python register_tool.py subflows/make_salt_water.json

## register all the subflows to DB
python register_tool.py subflows/


# delete DB
rm -rf workflow_db


# view DB
python check_db.py
