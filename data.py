import pandas as pd
import json

rows = []

with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)   
        
        n = len(data["messages"])
        for i in range(n):
            row = {
                "messages": data["messages"][i],
                "sender_labels": data["sender_labels"][i],
                "receiver_labels": data["receiver_labels"][i],
                "speakers": data["speakers"][i],
                "receivers": data["receivers"][i],
                "absolute_message_index": data["absolute_message_index"][i],
                "relative_message_index": data["relative_message_index"][i],
                "seasons": data["seasons"][i],
                "years": data["years"][i],
                "game_score": data["game_score"][i],
                "game_score_delta": data["game_score_delta"][i],
                "players": ", ".join(data["players"]),
                "game_id": data["game_id"]
            }
            rows.append(row)


df = pd.DataFrame(rows)


df.to_csv("train.csv", index=False, encoding="utf-8")

