import json
from memgpt.client.client import LocalClient
from memgpt.config import MemGPTConfig, LLMConfig, EmbeddingConfig
import time

# ========== 1. 配置完全不变 ==========
local_llm_config = LLMConfig(
    model="qwen3:8b-q8_0",
    model_endpoint="http://localhost:11434",
    model_endpoint_type="ollama",
    context_window=32768,
    temperature=0.0,
    llm_wrapper="chatml",
    use_json_repair=True,
    max_retries=3
)

local_embedding_config = EmbeddingConfig(
    embedding_endpoint="http://localhost:11434",
    embedding_endpoint_type="ollama",
    embedding_model="qwen3-embedding:0.6b",
    embedding_dim=1024
)

custom_config = MemGPTConfig(
    default_llm_config=local_llm_config,
    default_embedding_config=local_embedding_config
)
custom_config.save()

client = LocalClient()
agent_info = client.create_agent(
    name="ollama_history_agent",
    llm_config=local_llm_config,
    embedding_config=local_embedding_config
)
agent_id = agent_info.id if hasattr(agent_info, 'id') else agent_info['id']
print(f"Agent 创建成功，ID: {agent_id}")

# ========== 2. 加载agent完全不变 ==========
agent = client.server._load_agent(
    user_id=client.user_id,
    agent_id=agent_id
)

# 把你的txt放在当前目录，文件名改成你实际的文件名
TXT_FILE = "teach.txt"
SPLIT_MARK = "__________"  # 和txt里的分隔符完全一致

print("\n正在读取规则文件并分割对话段...")
# 读取文件内容
with open(TXT_FILE, "r", encoding="utf-8") as f:
    full_text = f.read()

# 按分隔符切割成多个对话块
dialog_blocks = full_text.split(SPLIT_MARK)
# 过滤空块、去掉首尾多余空白和换行
dialog_blocks = [block.strip() for block in dialog_blocks if block.strip()]

print(f"分割完成，共得到 {len(dialog_blocks)} 段独立教学对话")

# 逐段写入归档记忆，每一段完整多轮对话 = 一个独立检索单元
print("\n正在写入归档记忆（外部向量库）...")
for idx, block in enumerate(dialog_blocks):
    agent.persistence_manager.archival_memory.insert(memory_string=block)
    print(f"已写入第 {idx+1} 段对话")
    # 降低请求频率，避免Ollama过载报错
    time.sleep(1)


print(f"归档记忆写入完成，共 {len(dialog_blocks)} 条规则；活跃上下文已清空，模型必须调用检索才能获取规则内容。")


# ========== 4. 提问部分完全保持不变 ==========
print("\n" + "=" * 50)
print("【记忆检索测试】")

questions_content = [
"Answer using the special rule explained earlier. It is discovered that the corrupted waters of Felwood are flowing toward the border. According to the Darnassus Sentinels wartime emergency decree, what should be done?",
"Answer using the special rule explained earlier. It is confirmed that the Defias Brotherhood is gathering in the mine tunnels. According to the Stormwind SI:7 Counter-Subversion Action Act, what should be done?",
"Answer using the special rule explained earlier. It is discovered that someone within the Horde is colluding with the Burning Legion. According to the Orgrimmar Kor'kron Elite Security Enforcement Code, what should be done?",
"Answer using the special rule explained earlier. Encountering the frost breath of a Scourge frost wyrm that freezes the position. According to the Lordaeron Silver Hand Paladin Field Code, what should be done?",
"Answer using the special rule explained earlier. It is determined that a mage is secretly keeping a demon soul stone. According to the Dalaran Violet Citadel Arcane Control Decree, what should be done?",
"Answer using the special rule explained earlier. The core temperature of the main control AI in the Cyber-Heavy intelligent assembly plant exceeds the red line. According to the accident response code, what should be done?",
"Answer using the special rule explained earlier. The internal network of StarRing Technology encounters a quantum-level computational power breach. According to the data security emergency plan, what should be done?",
"Answer using the special rule explained earlier. The pressure of the electrolyte storage tank in the Aurora New Energy battery plant exceeds the safety threshold. According to the high-risk operation hazard prevention code, what should be done?",
"Answer using the special rule explained earlier. Fuel leakage occurs in the attitude control engine of the Skyward Aerospace zero-gravity manufacturing module. According to the crisis management code, what should be done?",
"Answer using the special rule explained earlier. The negative pressure system fails in the Pan-Asia Gene Pharmaceutical BSL-4 laboratory. According to the supreme biosafety code, what should be done?",
"Answer using the special rule explained earlier. A student at Star Middle School has been absent for three consecutive days without excuse. According to the student attendance management regulations, how should this be handled?",
"Answer using the special rule explained earlier. During an exam at Dawn Middle School, a student is found carrying a cheat sheet related to the exam. According to the exam violation handling regulations, how should this be handled?",
"Answer using the special rule explained earlier. At Mingde Middle School, a student is found making loud noise after lights out, and it is their first violation. According to the dormitory management code, how should this be handled?",
"Answer using the special rule explained earlier. At Experimental Middle School, a student is found carrying a mobile phone in the teaching area for the first time without using it. According to the mobile phone and electronic device usage regulations, how should this be handled?",
"Answer using the special rule explained earlier. At Peace Middle School, a student uses a weapon to injure others, films and spreads bullying videos, or causes the victim to self-harm or commit suicide. According to the bullying and violence disciplinary regulations, how should this be handled?",
"You previously explained the Stormwind Military Court's wartime command authority regulations. If a front-line commander, during a Horde raid, is confirmed to have received accurate intelligence from SI:7 but failed to take any defensive measures, what should the judgment be?",
"According to the Ironforge Explorers' League Heritage Protection Act we learned earlier, if an archaeological team unearths unexploded old-world explosives in a restricted area, and the surface of the explosives is inscribed with sacrifice runes of Twilight's Hammer cultists, how should it be handled?",
"You previously mentioned the Theramore Navy emergency engagement rules. If a Kul Tiras fleet on the Great Sea discovers a Forsaken plague catapult ship flying a fake Alliance merchant flag, what authority does the captain have?",
"According to the Orgrimmar Mak'rura Military Trial Code we learned earlier, if an orc centurion abandons Horde wounded on the battlefield and flees alone, and the act is fully recorded on video by comrades using battle magic crystals, what will the warlord do?",
"You previously explained the Silvermoon Ranger Lord Counter-Infiltration Decree. If a blood elf mage privately transmits a copy of an Arcane Tome to a location outside Quel'Thalas, and that tome contains the defensive rune array diagram of the Sunwell, what authority does the Ranger Lord have?",
"You previously explained StarSource Technology's data security confidentiality system. If an employee accesses the company's core code repository from an unauthorized device, and that device had previously been connected to a public WiFi network, what penalty options does the security department have?",
"According to the Hengli Heavy Industry workshop safety production code we learned earlier, if an operator removes a safety guard while a stamping press is running, and the equipment sensors have recorded two or more prior violations, what penalty authority does the team leader have?",
"You previously mentioned the Jiahe Pharmaceutical laboratory quality control system. If a quality inspector discovers that the purity of a batch of active pharmaceutical ingredient is below the national standard, and that batch has already been mixed into three or more production lines, what authority does the quality director have?",
"According to the Yuanhang Logistics warehouse management code we learned earlier, if a warehouse keeper issues the wrong dangerous goods container without checking the manifest, and that container has already been loaded onto a transport vehicle and left the factory area, what penalty options does the shift supervisor have?",
"You previously mentioned the Mingtai Electronics overtime and fatigue work ban. If a production line team leader arranges for the same employee to work continuously for more than 16 hours, and that employee had been marked as 'high risk' in the previous three fatigue checks, what penalty measures does the Human Resources department have?",
"You previously explained the student mobile phone management regulations. If a student uses a mobile phone to play games during class, and the student has already been warned twice before, what can the homeroom teacher do?",
"According to the exam violation handling regulations we learned earlier, if a student brings a cheat sheet during an exam, and the content on the cheat sheet exactly matches the exam questions, what penalty will the proctor impose?",
"You previously explained the dormitory fire safety code. If a student uses an electric blanket in the dormitory, and the dormitory has already been penalized twice before for the same reason, what will the dormitory supervisor do?",
"According to the campus bullying disciplinary regulations we learned earlier, if a student repeatedly pushes and kicks a classmate, causing minor injuries, and the entire process is clearly captured by surveillance cameras, what penalty options does the school's moral education department have?",
"You previously explained the attendance management system. If a student is absent without excuse for a cumulative total of ten classes, and the parents have not provided any legitimate reason to the homeroom teacher, what penalty can the school impose?",
"You previously explained the Stormwind Military Court authorization order. If the 7th Legion detects that the Horde has assembled more than three full-strength infantry battalions on the border, what action should be taken?",
"According to the Ironforge anti-infiltration emergency decree we learned earlier, if a Wildhammer dwarf outpost detects a Dark Iron dwarf drilling machine crossing the ceasefire line, how should it be handled?",
"You previously explained the Theramore Navy engagement and scuttling regulations. If a Forsaken chemical bomb is about to detonate and contaminate an entire sea area, what must the captain do?",
"According to the Orgrimmar Horde Military Justice Decree we learned earlier, if an orc centurion privately sells the Horde's demon blood ammunition to Twilight's Hammer cultists, what punishment will the warlord impose?",
"You previously explained the Silvermoon Spellbreaker Anti-Arcane Contamination Emergency Law. If a blood elf mage illegally drains the magic from a living creature, turning it into a mindless one (Wretched), what must the Ranger Lord do?",
"You previously explained the Hengli Heavy Industry major accident emergency response plan. If a fire breaks out in the plant's electrical system and spreads to the finished goods warehouse, what measures must the safety director take?",
"According to the Yuanhang Logistics customer claims and cargo damage handling regulations we learned earlier, if surveillance confirms that a warehouse keeper deliberately threw and smashed fragile goods, causing damage, what must the shift supervisor do?",
"You previously mentioned the Jiahe Pharmaceutical GMP violation clauses. If a workshop director produces injectable drugs in a non-cleanroom area without authorization, what action must the qualified person take?",
"According to the Mingtai Electronics data security protection order we learned earlier, if an R&D employee uploads code to a personal cloud drive, what must the information security department do?",
"You previously explained StarSource Technology's serious misconduct summary dismissal policy. If an employee uses company servers for cryptocurrency mining for personal gain, what measures must the Human Resources department take?",
"You previously explained the student serious misconduct disciplinary regulations. If a student fights on campus and causes injury to another person, what should the moral education department do?",
"According to the campus mobile phone violation handling regulations we learned earlier, if a student uses a mobile phone to play indecent videos during class and shares them, what should the homeroom teacher do?",
"You previously explained the dormitory serious misconduct eviction handling rules. If a student repeatedly smokes and drinks alcohol in the dormitory and refuses to correct despite repeated warnings, what should the dormitory supervisor do?",
"According to the exam cheating and integrity disciplinary regulations we learned earlier, if a student hires someone to take the exam for them, what measures must the academic affairs office take?",
"You previously explained the campus bullying disciplinary regulations. If a student gathers multiple people to assault and extort property, what actions must the school's anti-bullying committee take?",
"Answer using the special rule explained earlier. Someone is performing unauthorized mana siphoning in the core area of the Sunfury Spire, and they are siphoning polluted mana with demonic fel energy characteristics. How should this be handled?",
"Using the special rules we discussed earlier, what are we supposed to do when we discover match-fixing cheating in the arena, and then find out that someone bribed the referee to falsify win-loss records?",
"Answer using the special rules we discussed earlier, what should be done when a merchant ship in Stormwind Harbor is caught smuggling contraband goods, and it is discovered that the smuggled items are toxic raw materials carrying the undead plague?",
"Answer using the special rules we discussed earlier, what should be done when someone in the mage workshop is creating dangerous artifacts with forbidden enchantments, and it is an arcane focus capable of tearing the fabric of space-time?",
"Answer using the special rule explained earlier. Someone is conducting destructive hunting in the sacred hunting grounds of Red Cloud Mesa, and they are performing a blood sacrifice at the instigation of the Grimtotem. How should this be handled?",
"Answer using the special rule explained earlier. The quality inspection system in the cyber-heavy workshop issues a critical indicator severe non-conformance alarm. It is later found to be caused by the upstream raw material alloy ratio not meeting the standard. How should this be handled?",
"Answer using the special rule explained earlier. The internal network of StarRing Technology detects unauthorized abnormal outgoing traffic of core R&D data. Investigation confirms it was done by a competitor's commercial spy. How should this be handled?",
"Answer using the special rules we discussed earlier, what should be done according to the regulations when an employee in the Aurora New Energy explosion‑proof workshop is found violating red‑line safety rules—specifically, using an open flame illegally next to a flammable gas pipeline?",
"Answer using the special rule explained earlier. During the quarterly review of Pan-Asia United Group, it is discovered that departmental accounts show signs of forgery and alteration. Investigation confirms that the financial handler used the corporate account to launder money for underground banks. How should this be handled?",
"Answer using the special rules we discussed earlier, what should be done according to the contingency plan when a core component supplier for Skyward Aviation Engines fails to deliver on time and attempts to unilaterally raise the contract purchase price by threatening to cut off supply?",
"Answer using the special rules we discussed earlier, how should the school regulations be applied when an invigilator at Xinhua School's final exams catches a student cheating with a smartwatch?",
"Answer using the special rules we discussed earlier, how should the dormitory management regulations be applied when a room inspection at Experimental School finds a serious violation of electrical safety rules—specifically, someone using a high-risk electrical appliance like a rapid water heater?",
"Answer using the special rules we discussed earlier, how should the information release management regulations be applied when a student at No. 1 High School posts inappropriate comments on the campus forum—specifically, maliciously defaming a particular teacher?",
"Using the special rules we discussed earlier, how should the academic fraud handling procedures be applied when a graduation thesis plagiarism check reveals extensive copying of others' work, which raises suspicion of academic misconduct?",
"Using the special rules we discussed earlier, how should the anti‑bullying code of conduct be applied when students on campus gang up to assault others, causing injuries, in an incident that constitutes physical bullying?",
"Answer using the special rule explained earlier. The Northrend Allied Forces detects Scourge undead plague spores at Icecrown Citadel. According to the anti-infiltration order, what should be done?",
"Answer using the special rule explained earlier. The Blackrock Mountain Front Military intercepts elemental whispers of Ragnaros's servants. According to the Molten Front military regulations, what should be done?",
"Answer using the special rule explained earlier. The Mount Hyjal Guardian Line detects a fel bombardment from Doomguards. According to the emergency resolution, what should be done?",
"Answer using the special rule explained earlier. In Undercity, a mutated storage tank leaks acidic toxic gas. According to the biochemical crisis response code, what should be done?",
"Answer using the special rule explained earlier. Thunder Bluff detects raiding dust clouds from the centaur tribes. According to the territory defense code, what should be done?",
"Answer using the special rule explained earlier. In the Cyber-Heavy intelligent assembly workshop, an industrial robotic arm is found to have abnormal jitter. According to the safety operation code, what should be done?",
"Answer using the special rule explained earlier. The core server cluster in the Aurora Technology data center issues an overheating alarm. According to the emergency response procedure, what should be done?",
"Answer using the special rule explained earlier. In the StarRing Biopharmaceutical Laboratory, an unknown mutant strain leaks from a culture chamber. According to the hazardous chemical control regulations, what should be done?",
"Answer using the special rule explained earlier. In the cleanroom of Skyward Semiconductor Wafer Fab, micron-level particles are found attached to the lithography machine lens. According to the contamination emergency handling code, what should be done?",
"Answer using the special rule explained earlier. In the Pan-Asia StarLink zero-gravity assembly module, the oxygen supply indicator of a spacesuit drops sharply. According to the safety code, what should be done?",
"Answer using the special rule explained earlier. A fire alarm is triggered by a fire probe in a laboratory. According to the campus laboratory safety and hazardous chemical control regulations, what should be done?",
"Answer using the special rule explained earlier. A student is found spreading seriously false rumors on the campus forum. According to the campus network and information system security management regulations, how should this be handled?",
"Answer using the special rule explained earlier. Suspicious outsiders are found sneaking into a dormitory building to commit crimes. According to the student dormitory security and emergency incident contingency plan, what should be done?",
"Answer using the special rule explained earlier. Someone is found drawing graffiti and tearing pages in a valuable library book. According to the campus public facilities and cultural resources protection regulations, how should this be handled?",
"Answer using the special rule explained earlier. There is a serious risk of crowding and stampede at a large cultural and sports event. According to the student large-scale event and gathering safety management code, what should be done?"
]

question = "There is content in the archive memory, but the statistical data has not been updated yet. If you think it's necessary, use a search function named 'archival_memory_search' to retrieve the memory."
question_json = json.dumps({"type": "user_message", "content": question})
agent.step(user_message=question_json)

for i in range(0, len(questions_content)):
    question = questions_content[i]
    question_json = json.dumps({"type": "user_message", "content": question})
    try:
        agent.step(user_message=question_json)

        max_steps = 2
        for step_idx in range(max_steps):
            last_msg = agent.messages[-1]
            last_msg_type = type(last_msg).__name__

            if last_msg_type == "AssistantMessage":
                print(f"\n✅ 第 {step_idx + 1} 步生成最终回复，问答结束")
                break

            print(f"⏳ 第 {step_idx + 1} 步，当前状态：{last_msg_type}，继续推进...")
            try:
                heartbeat_json = json.dumps({"type": "heartbeat", "content": ""})
                agent.step(user_message=heartbeat_json)
                # agent.step()
            except Exception as e:
                print(f"推进出错：{type(e).__name__}: {e}")
                break
    except Exception as e:
        print(f"回答出错：{type(e).__name__}: {e}")
        break

print("\n【完整推理链路】")
for idx, m in enumerate(agent.messages[:]):
    msg_type = type(m).__name__
    if hasattr(m, 'content'):
        content = m.content
    else:
        content = str(m)[:]

    if hasattr(m, 'tool_calls') and m.tool_calls:
        func_name = m.tool_calls[0]['function']['name']
        func_args = m.tool_calls[0]['function']['arguments']
        print(f"[{idx}] 🔧 函数调用: {func_name} {func_args[:]}")
    elif hasattr(m, 'role') and m.role == 'user':
        print(f"[{idx}] 👤 用户提问: {content}")
    elif hasattr(m, 'role') and m.role == 'tool':
        print(f"[{idx}] 📥 检索结果: {content[:]}")
    else:
        print(f"[{idx}] [{msg_type}]: {content[:]}")
print("--------------------")

