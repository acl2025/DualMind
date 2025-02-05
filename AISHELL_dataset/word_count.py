import os

def count_text_stats(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        lines = content.split('\n')
        words = content.split()
        characters = len(content)
        
    return {
        'characters': characters,
        'words': len(words),
        'lines': len(lines)
    }

def process_directory(directory_path):
    total_stats = {'characters': 0, 'words': 0, 'lines': 0}
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            file_stats = count_text_stats(file_path)
            print(f"{filename}:")
            print(f"  Characters: {file_stats['characters']}")
            print(f"  Words: {file_stats['words']}")
            print(f"  Lines: {file_stats['lines']}")
            print()
            
            # 累加总计
            for key in total_stats:
                total_stats[key] += file_stats[key]
    
    return total_stats

# 指定要处理的目录
directory_path = "/home/leon/agent/AISHELL_dataset/test/text_raw"

total_stats = process_directory(directory_path)

print("Total stats for all files:")
print(f"  Total Characters: {total_stats['characters']}")
print(f"  Total Words: {total_stats['words']}")
print(f"  Total Lines: {total_stats['lines']}")