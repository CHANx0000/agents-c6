'''
manager

tools
sub agent 1
sub agent 2
sub agent 3

sense(goal,iteration)
context={goal, iteration, memory[history] last 3, memory[notes]}
return Context


plan(context)

system_prompt = '
Read the context and give me a response similar to expected output
You are a manager. You have tools(specialized agents).
tools = tools_list
Please dont use own facts. Relay on sub-agents for information.
expected output [JSON]
decision={
    action: complete or tool/sub-agent()
    args: if action = tool/sub-agent()
    reasoning: ?
    answer: if complete, final answer else null
}'
user_prompt = 'context, what next?'

llm call = system_prompt
output = response
decision = {
}
return decision


Act(decision)
if decision = tool/sub-agent()
    call tool/sub-agent with args
    result = tool/sub-agent(args)
    return result
else if decision = complete
    return result = complete


observe(result, decision)
if result.subagent has error
    observation= [kind: error, message: result.error,ok: false]
elif result.subagent has output
    observation = [kind: output, message: result.output, ok: true]
else decision.action = complete
    observation = [kind: complete, message: result.complete, ok: true]
return observation


Reflect(goal, decision, observation)
if decision.action = complete
    return true
else
    memory[history].append(tool_used: decision.tool, args: decision.args, reasoning: decision.reasoning, ok: observation[ok])
    return false

'''
