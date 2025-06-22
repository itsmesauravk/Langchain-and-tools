
from langchain.tools import tool
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing  import Annotated
import json
from dotenv import load_dotenv
load_dotenv()



# TOOL CREATION
@tool
def get_currency_conversion_rate(base_currency:str, target_currency:str) -> float:
    """Get the conversion rate from base_currency to target_currency."""
    
    url =f"https://v6.exchangerate-api.com/v6/f0aa50e2a524b7090b473fd4/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    return response.json()


# print(get_currency_conversion_rate({"base_currency": "NPR", "target_currency": "EUR"}))

@tool
def convert_currency(base_currency_value: float, conversation_rate:Annotated[float, InjectedToolArg]) -> float:
    """Convert a value from base_currency to target_currency using the conversion rate."""
    
    return base_currency_value * conversation_rate


# print(convert_currency({'base_currency_value':1400, 'conversation_rate':159.6316}))   #1400.00, 159.6316


# Tool binding
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=500)


llm_with_tools = llm.bind_tools([get_currency_conversion_rate, convert_currency])  # now llm can use the tools


query = HumanMessage("what is conversion rate from EUR to  NPR, and can you convert 100 EUR to NPR?")  # create a query for the llm

messages = [query]  # create a list of messages with the query

llm_response = llm_with_tools.invoke(messages)  # invoke the llm with the messages
messages.append(llm_response)  # append the llm response to the messages

# print(llm_response.tool_calls)  # print the llm response
# [{'name': 'get_currency_conversion_rate', 'args': {'base_currency': 'EUR', 'target_currency': 'NPR'}, 'id': '12be5c54-5e1d-4580-8c8f-efc6b942e55e', 'type': 'tool_call'}, {'name': 'convert_currency', 'args': {'base_currency_value': 100.0}, 'id': '6abc275a-5d89-492c-abfd-390c14f5c87a', 'type': 'tool_call'}]

for tool_call in llm_response.tool_calls:
    #execute the first tool call to get the conversion rate
    if tool_call['name'] == "get_currency_conversion_rate":
        #fetch the conversion rate from the first tool call
        conversion_rate_response = get_currency_conversion_rate.invoke(tool_call)
        # print(json.loads(conversion_rate_response.content))
        conversion_rate = json.loads(conversion_rate_response.content)['conversion_rate']
        # print(conversion_rate)
        messages.append(conversion_rate_response)

    #execute the second tool call to convert the currency getting conversion rate from the first tool call
    if tool_call['name'] == "convert_currency":
        #fetch the current 
        tool_call['args']['conversation_rate'] = conversion_rate

        converted_currency_response = convert_currency.invoke(tool_call)

        messages.append(converted_currency_response)




# print(messages)

# FINAL RESPONSE
final_response = llm_with_tools.invoke(messages)  # invoke the llm with the messages again to get the final response

print(final_response.content)  # print the final response from the llm