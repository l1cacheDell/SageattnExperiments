from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from rich import print
from sageattention import sageattn, sageattn_varlen
import torch.nn.functional as F

# model_name = "/root/paddlejob/workspace/env_run/output/dongyazhu/weight/Llama-2-7b-chat-hf"  # 使用适当的模型名称
model_name = "/root/paddlejob/workspace/env_run/output/dongyazhu/weight/Qwen2.5-7B-Instruct"
input_texts = ["""Five score years ago, a great American, in whose symbolic shadow we stand today, 
signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to 
millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous 
daybreak to end the long night of their captivity. But 100 years later, the Negro still is not free. 
One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation 
and the chains of discrimination. One hundred years later, the Negro lives on a lonely island of 
poverty in the midst of a vast ocean of material prosperity. One hundred years later the Negro is still 
languished in the corners of American society and finds himself in exile in his own land. And so we've come 
here today to dramatize a shameful condition. In a sense we've come to our nation's capital to cash a check. When the architects of our republic wrote the magnificent words of the Constitution and the Declaration of Independence, they were signing a promissory note to which every American was to fall heir. This note was a promise that all men — yes, Black men as well as white men — would be guaranteed the unalienable rights of life, liberty and the pursuit of happiness.It is obvious today that America has defaulted on this promissory note insofar as her citizens of color are concerned. Instead of honoring this sacred obligation, America has given the Negro people a bad check, a check which has come back marked insufficient funds. We refuse to believe that there are insufficient funds in the great vaults of opportunity of this nation. And so we've come to cash this check, a check that will give us upon demand the riches of freedom and the security of justice.

We have also come to this hallowed spot to remind America of the fierce urgency of now. This is no time to engage in the luxury of cooling off or to take the tranquilizing drug of gradualism.

Now is the time to make real the promises of democracy. Now is the time to rise from the dark and desolate valley of segregation to the sunlit path of racial justice. Now is the time to lift our nation from the quick sands of racial injustice to the solid rock of brotherhood. Now is the time to make justice a reality for all of God's children.It would be fatal for the nation to overlook the urgency of the moment. This sweltering summer of the Negro's legitimate discontent will not pass until there is an invigorating autumn of freedom and equality. 1963 is not an end, but a beginning. Those who hope that the Negro needed to blow off steam and will now be content will have a rude awakening if the nation returns to business as usual.

There will be neither rest nor tranquility in America until the Negro is granted his citizenship rights. The whirlwinds of revolt will continue to shake the foundations of our nation until the bright day of justice emerges.

But there is something that I must say to my people who stand on the warm threshold which leads into the palace of justice. In the process of gaining our rightful place, we must not be guilty of wrongful deeds. Let us not seek to satisfy our thirst for freedom by drinking from the cup of bitterness and hatred.We must forever conduct our struggle on the high plane of dignity and discipline. We must not allow our creative protest to degenerate into physical violence. Again and again, we must rise to the majestic heights of meeting physical force with soul force. The marvelous new militancy which has engulfed the Negro community must not lead us to a distrust of all white people, for many of our white brothers, as evidenced by their presence here today, have come to realize that their destiny is tied up with our destiny.And they have come to realize that their freedom is inextricably bound to our freedom. We cannot walk alone. And as we walk, we must make the pledge that we shall always march ahead. We cannot turn back.

There are those who are asking the devotees of civil rights, when will you be satisfied? We can never be satisfied as long as the Negro is the victim of the unspeakable horrors of police brutality. We can never be satisfied as long as our bodies, heavy with the fatigue of travel, cannot gain lodging in the motels of the highways and the hotels of the cities.

No, no, we are not satisfied, and we will not be satisfied until justice rolls down like waters, and righteousness like a mighty stream.I am not unmindful that some of you have come here out of great trials and tribulations. Some of you have come fresh from narrow jail cells. Some of you have come from areas where your quest for freedom left you battered by the storms of persecution and staggered by the winds of police brutality. You have been the veterans of creative suffering. Continue to work with the faith that unearned suffering is redemptive. Go back to Mississippi, go back to Alabama, go back to South Carolina, go back to Georgia, go back to Louisiana, go back to the slums and ghettos of our Northern cities, knowing that somehow this situation can and will be changed.Let us not wallow in the valley of despair, I say to you today, my friends.

So even though we face the difficulties of today and tomorrow, I still have a dream. It is a dream deeply rooted in the American dream. I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident, that all men are created equal. I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood.

I have a dream that one day even the state of Mississippi, a state sweltering with the heat of injustice, sweltering with the heat of oppression will be transformed into an oasis of freedom and justice.
Five score years ago, a great American, in whose symbolic shadow we stand today, 
signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to 
millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous 
daybreak to end the long night of their captivity. But 100 years later, the Negro still is not free. 
One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation 
and the chains of discrimination. One hundred years later, the Negro lives on a lonely island of 
poverty in the midst of a vast ocean of material prosperity. One hundred years later the Negro is still 
languished in the corners of American society and finds himself in exile in his own land. And so we've come 
here today to dramatize a shameful condition. In a sense we've come to our nation's capital to cash a check. When the architects of our republic wrote the magnificent words of the Constitution and the Declaration of Independence, they were signing a promissory note to which every American was to fall heir. This note was a promise that all men — yes, Black men as well as white men — would be guaranteed the unalienable rights of life, liberty and the pursuit of happiness.It is obvious today that America has defaulted on this promissory note insofar as her citizens of color are concerned. Instead of honoring this sacred obligation, America has given the Negro people a bad check, a check which has come back marked insufficient funds. We refuse to believe that there are insufficient funds in the great vaults of opportunity of this nation. And so we've come to cash this check, a check that will give us upon demand the riches of freedom and the security of justice.

We have also come to this hallowed spot to remind America of the fierce urgency of now. This is no time to engage in the luxury of cooling off or to take the tranquilizing drug of gradualism.

Now is the time to make real the promises of democracy. Now is the time to rise from the dark and desolate valley of segregation to the sunlit path of racial justice. Now is the time to lift our nation from the quick sands of racial injustice to the solid rock of brotherhood. Now is the time to make justice a reality for all of God's children.It would be fatal for the nation to overlook the urgency of the moment. This sweltering summer of the Negro's legitimate discontent will not pass until there is an invigorating autumn of freedom and equality. 1963 is not an end, but a beginning. Those who hope that the Negro needed to blow off steam and will now be content will have a rude awakening if the nation returns to business as usual.

There will be neither rest nor tranquility in America until the Negro is granted his citizenship rights. The whirlwinds of revolt will continue to shake the foundations of our nation until the bright day of justice emerges.

But there is something that I must say to my people who stand on the warm threshold which leads into the palace of justice. In the process of gaining our rightful place, we must not be guilty of wrongful deeds. Let us not seek to satisfy our thirst for freedom by drinking from the cup of bitterness and hatred.We must forever conduct our struggle on the high plane of dignity and discipline. We must not allow our creative protest to degenerate into physical violence. Again and again, we must rise to the majestic heights of meeting physical force with soul force. The marvelous new militancy which has engulfed the Negro community must not lead us to a distrust of all white people, for many of our white brothers, as evidenced by their presence here today, have come to realize that their destiny is tied up with our destiny.And they have come to realize that their freedom is inextricably bound to our freedom. We cannot walk alone. And as we walk, we must make the pledge that we shall always march ahead. We cannot turn back.

There are those who are asking the devotees of civil rights, when will you be satisfied? We can never be satisfied as long as the Negro is the victim of the unspeakable horrors of police brutality. We can never be satisfied as long as our bodies, heavy with the fatigue of travel, cannot gain lodging in the motels of the highways and the hotels of the cities.

No, no, we are not satisfied, and we will not be satisfied until justice rolls down like waters, and righteousness like a mighty stream.I am not unmindful that some of you have come here out of great trials and tribulations. Some of you have come fresh from narrow jail cells. Some of you have come from areas where your quest for freedom left you battered by the storms of persecution and staggered by the winds of police brutality. You have been the veterans of creative suffering. Continue to work with the faith that unearned suffering is redemptive. Go back to Mississippi, go back to Alabama, go back to South Carolina, go back to Georgia, go back to Louisiana, go back to the slums and ghettos of our Northern cities, knowing that somehow this situation can and will be changed.Let us not wallow in the valley of despair, I say to you today, my friends.

So even though we face the difficulties of today and tomorrow, I still have a dream. It is a dream deeply rooted in the American dream. I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident, that all men are created equal. I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood.

I have a dream that one day even the state of Mississippi, a state sweltering with the heat of injustice, sweltering with the heat of oppression will be transformed into an oasis of freedom and justice.
Five score years ago, a great American, in whose symbolic shadow we stand today, 
signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to 
millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous 
daybreak to end the long night of their captivity. But 100 years later, the Negro still is not free. 
One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation 
and the chains of discrimination. One hundred years later, the Negro lives on a lonely island of 
poverty in the midst of a vast ocean of material prosperity. One hundred years later the Negro is still 
languished in the corners of American society and finds himself in exile in his own land. And so we've come 
here today to dramatize a shameful condition. In a sense we've come to our nation's capital to cash a check. When the architects of our republic wrote the magnificent words of the Constitution and the Declaration of Independence, they were signing a promissory note to which every American was to fall heir. This note was a promise that all men — yes, Black men as well as white men — would be guaranteed the unalienable rights of life, liberty and the pursuit of happiness.It is obvious today that America has defaulted on this promissory note insofar as her citizens of color are concerned. Instead of honoring this sacred obligation, America has given the Negro people a bad check, a check which has come back marked insufficient funds. We refuse to believe that there are insufficient funds in the great vaults of opportunity of this nation. And so we've come to cash this check, a check that will give us upon demand the riches of freedom and the security of justice.

We have also come to this hallowed spot to remind America of the fierce urgency of now. This is no time to engage in the luxury of cooling off or to take the tranquilizing drug of gradualism.

Now is the time to make real the promises of democracy. Now is the time to rise from the dark and desolate valley of segregation to the sunlit path of racial justice. Now is the time to lift our nation from the quick sands of racial injustice to the solid rock of brotherhood. Now is the time to make justice a reality for all of God's children.It would be fatal for the nation to overlook the urgency of the moment. This sweltering summer of the Negro's legitimate discontent will not pass until there is an invigorating autumn of freedom and equality. 1963 is not an end, but a beginning. Those who hope that the Negro needed to blow off steam and will now be content will have a rude awakening if the nation returns to business as usual.

There will be neither rest nor tranquility in America until the Negro is granted his citizenship rights. The whirlwinds of revolt will continue to shake the foundations of our nation until the bright day of justice emerges.

But there is something that I must say to my people who stand on the warm threshold which leads into the palace of justice. In the process of gaining our rightful place, we must not be guilty of wrongful deeds. Let us not seek to satisfy our thirst for freedom by drinking from the cup of bitterness and hatred.We must forever conduct our struggle on the high plane of dignity and discipline. We must not allow our creative protest to degenerate into physical violence. Again and again, we must rise to the majestic heights of meeting physical force with soul force. The marvelous new militancy which has engulfed the Negro community must not lead us to a distrust of all white people, for many of our white brothers, as evidenced by their presence here today, have come to realize that their destiny is tied up with our destiny.And they have come to realize that their freedom is inextricably bound to our freedom. We cannot walk alone. And as we walk, we must make the pledge that we shall always march ahead. We cannot turn back.

There are those who are asking the devotees of civil rights, when will you be satisfied? We can never be satisfied as long as the Negro is the victim of the unspeakable horrors of police brutality. We can never be satisfied as long as our bodies, heavy with the fatigue of travel, cannot gain lodging in the motels of the highways and the hotels of the cities.

No, no, we are not satisfied, and we will not be satisfied until justice rolls down like waters, and righteousness like a mighty stream.I am not unmindful that some of you have come here out of great trials and tribulations. Some of you have come fresh from narrow jail cells. Some of you have come from areas where your quest for freedom left you battered by the storms of persecution and staggered by the winds of police brutality. You have been the veterans of creative suffering. Continue to work with the faith that unearned suffering is redemptive. Go back to Mississippi, go back to Alabama, go back to South Carolina, go back to Georgia, go back to Louisiana, go back to the slums and ghettos of our Northern cities, knowing that somehow this situation can and will be changed.Let us not wallow in the valley of despair, I say to you today, my friends.

So even though we face the difficulties of today and tomorrow, I still have a dream. It is a dream deeply rooted in the American dream. I have a dream that one day this nation will rise up and live out the true meaning of its creed: We hold these truths to be self-evident, that all men are created equal. I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood.

I have a dream that one day even the state of Mississippi, a state sweltering with the heat of injustice, sweltering with the heat of oppression will be transformed into an oasis of freedom and justice.
Five score years ago, a great American, in whose symbolic shadow we stand today, 
signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to 
millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous 
daybreak to end the long night of their captivity. But 100 years later, the Negro still is not free. 
One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation 
and the chains of discrimination. One hundred years later, the Negro lives on a lonely island of 
poverty in the midst of a vast ocean of material prosperity. One hundred years later the Negro is still 
languished in the corners of American society and finds himself in exile in his own land. And so we've come 
here today to dramatize a shameful condition. In a sense we've come to our nation's capital to cash a check. When the architects of our republic wrote the magnificent words of the Constitution and the Declaration of Independence, they were signing a promissory note to which every American was to fall heir. This note was a promise that all men — yes, Black men as well as white men — would be guaranteed the unalienable rights of life, liberty and the pursuit of happiness.It is obvious today that America has defaulted on this promissory note insofar as her citizens of color are concerned. Instead of honoring this sacred obligation, America has given the Negro people a bad check, a check which has come back marked insufficient funds. We refuse to believe that there are insufficient funds in the great vaults of opportunity of this nation. And so we've come to cash this check, a check that will give us upon demand the riches of freedom and the security of justice.

We have also come to this hallowed spot to remind America of the fierce urgency of now. This is no time to engage in the luxury of cooling off or to take the tranquilizing drug of gradualism.
"""]

# input_texts = ["What is the meaning of life?"]

max_len_gen = 7000

def naive_model_inference():
    global model_name, input_texts, max_len_gen
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer.pad_token = tokenizer.eos_token
    # 输入文本

    print(f"using: {model}")

    # 编码输入文本
    inputs = tokenizer(input_texts, return_tensors="pt").to("cuda")

    # 生成输出
    start = time.monotonic()
    for i in range(5):
        outputs = model.generate(**inputs, max_length=max_len_gen, temperature=0.01, do_sample=True)

    end = time.monotonic()
    print(f"Naive inference took: {end - start} s")
    # 解码并打印输出
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(generated_text)

    return end - start


def flash_attn_inference():
    # 加载Llama2-7b模型和分词器
    global model_name, input_texts, max_len_gen
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", attn_implementation="flash_attention_2", torch_dtype="bfloat16")
    tokenizer.pad_token = tokenizer.eos_token
    # 输入文本

    print(f"using: {model}")

    # 编码输入文本
    inputs = tokenizer(input_texts, return_tensors="pt").to("cuda")
    print("input token length: ", inputs.input_ids.shape)

    # 生成输出
    start = time.monotonic()
    for i in range(5):
        outputs = model.generate(**inputs, min_length=max_len_gen, max_length=max_len_gen)

    end = time.monotonic()
    print(f"FA2 inference took: {end - start} s")

    # 解码并打印输出
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(generated_text)

    gen_words = len(generated_text[0])

    return (end - start) / gen_words


def sage_attn_inference():
    # 加载Llama2-7b模型和分词器
    global model_name, input_texts, max_len_gen
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    F.scaled_dot_product_attention = sageattn
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer.pad_token = tokenizer.eos_token
    # 输入文本

    print(f"using: {model}")

    # 编码输入文本
    inputs = tokenizer(input_texts, return_tensors="pt").to("cuda")
    print("input token length: ", inputs.input_ids.shape)

    # 生成输出
    start = time.monotonic()
    for i in range(5):
        outputs = model.generate(**inputs, min_length=max_len_gen, max_length=max_len_gen)

    end = time.monotonic()
    print(f"sage attn inference took: {end - start} s")

    # 解码并打印输出
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(generated_text)

    gen_words = len(generated_text[0])

    return (end - start) / gen_words



if __name__ == '__main__':
    # latency_1 = naive_model_inference()
    latency_2 = flash_attn_inference()
    torch.cuda.empty_cache()
    latency_sageattn = sage_attn_inference()
    batch_size = len(input_texts)
    # print(f"SDPA -> FA2      speed up is: {latency_1 / latency_2} x, batch_size: {batch_size}")
    print(f"FA2  -> SageAttn speed up is: {latency_2 / latency_sageattn} x, batch_size: {batch_size}")