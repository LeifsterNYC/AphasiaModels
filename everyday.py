"""Short everyday sentences of the kind a person with aphasia actually needs
to say: wants, pain, family, places, feelings. Subtitles and web text contain
almost none of these, and they are where short inputs come from."""
import random

THINGS = ["some water", "a glass of water", "some coffee", "a cup of tea", "some juice",
          "something to eat", "some food", "some soup", "some bread", "breakfast", "lunch",
          "dinner", "my medicine", "my pills", "my glasses", "my phone", "a blanket",
          "another pillow", "my coat", "my shoes", "the remote", "a newspaper", "my book",
          "some help", "a break", "some rest", "some fresh air", "a shower", "the bathroom",
          "my wallet", "my keys", "my cane", "my walker", "my hearing aid", "a tissue",
          "some milk", "an apple", "some fruit", "a sandwich", "some ice", "my sweater"]
ACTS = ["go home", "go to bed", "go outside", "go to the store", "go to the bathroom",
        "take a shower", "take my medicine", "take a nap", "eat something", "eat breakfast",
        "eat dinner", "drink some water", "watch TV", "watch the news", "read my book",
        "call my son", "call my daughter", "call my wife", "call my husband", "call the doctor",
        "see the doctor", "see my grandchildren", "sit down", "lie down", "stand up",
        "go for a walk", "walk the dog", "feed the cat", "make coffee", "cook dinner",
        "wash my hands", "brush my teeth", "get dressed", "change my shirt", "rest for a while",
        "listen to music", "talk to my sister", "visit my brother", "buy some bread",
        "buy milk", "pay the bill", "find my glasses", "open the window", "close the door",
        "turn off the light", "turn on the TV", "write a letter", "go to church", "go to work",
        "practice my speech", "do my exercises", "sleep"]
BODY = ["head", "back", "leg", "arm", "hand", "knee", "foot", "stomach", "neck", "shoulder",
        "chest", "throat", "hip", "tooth", "ear", "eye"]
PEOPLE = ["my wife", "my husband", "my son", "my daughter", "my brother", "my sister",
          "my friend", "my doctor", "the nurse", "my neighbor", "my mother", "my father",
          "my granddaughter", "my grandson", "the therapist", "my boss"]
FUTURE = ["tomorrow", "tomorrow morning", "tonight", "this afternoon", "on Sunday", "on Monday",
          "next week", "this weekend", "later", "soon", "on Friday", "this evening"]
PAST = ["yesterday", "last night", "this morning", "last week", "on Tuesday", "two days ago",
        "last Sunday", "earlier today"]
FEEL = ["tired", "hungry", "thirsty", "cold", "hot", "sick", "dizzy", "happy", "sad", "angry",
        "scared", "worried", "confused", "bored", "lonely", "better", "worse", "fine",
        "frustrated", "sleepy", "nervous", "okay"]
OBJECTS = ["glasses", "phone", "keys", "wallet", "book", "cane", "shoes", "coat", "pills",
           "hat", "watch", "purse", "remote", "hearing aid", "slippers", "bag"]
PLACES = ["store", "hospital", "doctor", "park", "bank", "pharmacy", "church", "restaurant",
          "post office", "library", "market", "gym", "clinic", "beach", "dentist"]
WEATHER = ["raining", "snowing", "sunny", "cold", "hot", "windy", "nice outside", "cloudy"]
ROOM = ["cold", "hot", "loud", "dark", "bright", "quiet"]
FOODS = ["the soup", "the food here", "this coffee", "fish", "the hospital food", "eggs",
         "this medicine", "the chicken", "tea", "vegetables", "this room", "the new nurse"]

TEMPLATES = [
    (4, lambda r: f"I want {r(THINGS)}."),
    (3, lambda r: f"I need {r(THINGS)}."),
    (2, lambda r: f"I would like {r(THINGS)}."),
    (2, lambda r: f"Can I have {r(THINGS)}?"),
    (2, lambda r: f"I do not want {r(THINGS)}."),
    (2, lambda r: f"I don't need {r(THINGS)} right now."),
    (4, lambda r: f"I want to {r(ACTS)}."),
    (3, lambda r: f"I need to {r(ACTS)}."),
    (2, lambda r: f"I want to {r(ACTS)} now."),
    (2, lambda r: f"I have to {r(ACTS)} {r(FUTURE)}."),
    (2, lambda r: f"I am going to {r(ACTS)} {r(FUTURE)}."),
    (3, lambda r: f"I do not want to {r(ACTS)}."),
    (3, lambda r: f"I cannot {r(ACTS)}."),
    (2, lambda r: f"I can't {r(ACTS)} today."),
    (2, lambda r: f"I did not {r(ACTS)} {r(PAST)}."),
    (2, lambda r: f"Can you help me {r(ACTS)}?"),
    (2, lambda r: f"Please help me {r(ACTS)}."),
    (1, lambda r: f"Do I have to {r(ACTS)}?"),
    (3, lambda r: f"My {r(BODY)} hurts."),
    (2, lambda r: f"My {r(BODY)} hurts today."),
    (2, lambda r: f"My {r(BODY)} hurt {r(PAST)}."),
    (2, lambda r: f"I have a pain in my {r(BODY)}."),
    (2, lambda r: f"My {r(BODY)} does not hurt anymore."),
    (1, lambda r: f"My {r(BODY)} hurts, so I need to {r(ACTS)}."),
    (3, lambda r: f"{r(PEOPLE).capitalize()} is coming {r(FUTURE)}."),
    (3, lambda r: f"{r(PEOPLE).capitalize()} will visit {r(FUTURE)}."),
    (2, lambda r: f"{r(PEOPLE).capitalize()} called me {r(PAST)}."),
    (2, lambda r: f"{r(PEOPLE).capitalize()} visited me {r(PAST)}."),
    (3, lambda r: f"{r(PEOPLE).capitalize()} is not coming {r(FUTURE)}."),
    (2, lambda r: f"{r(PEOPLE).capitalize()} did not call {r(PAST)}."),
    (2, lambda r: f"{r(PEOPLE).capitalize()} does not work {r(['today', 'on Sundays', 'anymore', 'this week'])}."),
    (2, lambda r: f"{r(['He', 'She'])} is not going to work {r(['today', 'tomorrow', 'this week'])}."),
    (2, lambda r: f"{r(['He', 'She', 'They', 'We'])} did not {r(ACTS).replace('my ', 'the ')} {r(PAST)}."),
    (2, lambda r: f"{r(PEOPLE).capitalize()} took me to the {r(PLACES)} {r(PAST)}."),
    (2, lambda r: f"{r(PEOPLE).capitalize()} told me to {r(ACTS)}."),
    (2, lambda r: f"{r(PAST).capitalize()} {r(PEOPLE)} told me to {r(ACTS)}."),
    (2, lambda r: f"I want to see {r(PEOPLE)}."),
    (2, lambda r: f"When is {r(PEOPLE)} coming?"),
    (1, lambda r: f"Where is {r(PEOPLE)}?"),
    (3, lambda r: f"I am {r(FEEL)}."),
    (2, lambda r: f"I feel {r(FEEL)} today."),
    (2, lambda r: f"I am very {r(FEEL)}."),
    (3, lambda r: f"I am not {r(FEEL)}."),
    (2, lambda r: f"I was {r(FEEL)} {r(PAST)}, but I feel {r(['better', 'fine', 'okay'])} now."),
    (3, lambda r: f"I cannot find my {r(OBJECTS)}."),
    (2, lambda r: f"I can't find my {r(OBJECTS)}."),
    (2, lambda r: f"Where is my {r(OBJECTS)}?"),
    (2, lambda r: f"I lost my {r(OBJECTS)}."),
    (1, lambda r: f"Have you seen my {r(OBJECTS)}?"),
    (3, lambda r: f"I went to the {r(PLACES)} {r(PAST)}."),
    (3, lambda r: f"I am going to the {r(PLACES)} {r(FUTURE)}."),
    (2, lambda r: f"I did not go to the {r(PLACES)} {r(PAST)}."),
    (2, lambda r: f"I want to go to the {r(PLACES)}."),
    (2, lambda r: f"I do not want to go to the {r(PLACES)}."),
    (2, lambda r: f"It is {r(WEATHER)} today."),
    (2, lambda r: f"It is {r(WEATHER)} today, so I cannot {r(ACTS)}."),
    (2, lambda r: f"It is too {r(ROOM)} in here."),
    (2, lambda r: f"I like {r(FOODS)}."),
    (3, lambda r: f"I do not like {r(FOODS)}."),
    (2, lambda r: f"I don't like {r(FOODS)}."),
    (2, lambda r: f"I need {r(THINGS)} in the morning."),
    (2, lambda r: f"I forgot to {r(ACTS)} {r(PAST)}."),
    (1, lambda r: f"I never {r(ACTS)} in the morning."),
    (2, lambda r: f"The doctor said I should {r(ACTS)}."),
    (2, lambda r: f"The doctor said I should not {r(ACTS)}."),
    (1, lambda r: f"What time is {r(['lunch', 'dinner', 'breakfast', 'my appointment', 'the doctor coming'])}?"),
    (1, lambda r: f"Thank you for {r(['helping me', 'coming', 'the food', 'your help', 'visiting me'])}."),
]


def everyday_sentences(n, seed=7):
    rng = random.Random(seed)
    weights = [w for w, _ in TEMPLATES]
    makers = [m for _, m in TEMPLATES]
    out = set()
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        out.add(rng.choices(makers, weights)[0](rng.choice))
    return sorted(out)


if __name__ == "__main__":
    for s in random.Random(1).sample(everyday_sentences(9000), 25):
        print(s)
