import argparse
from scripts import run

def main():
    parser = argparse.ArgumentParser(description='Run the main logic of the script.')
    
    # Add arguments as per the docstring of the run function
    parser.add_argument('--procedures', type=dict, default={
            'load_models': True,
            'crawl_wiktionary_nouns': False,
            'crawl_wiktionary_adjectives': False,
            'calculate_adjective_similarities': True,
            'select_top_adjectives': True,
            'remove_adjective_duplicates': True,
            'adjective_definition_review': True,
            'remove_unwanted_adjectives': True,
            'create_stimulus_files': True,
            'conduct_control_tests': True,
            'conduct_experimental_tests': True,
        }, help='Control flags for pipeline procedures. Keys:\n'
            '- \'load_models\' (bool): Load gensim models.\n'
            '- \'crawl_wiktionary_adjectives\' (bool): Crawl Wiktionary for adjectives.\n'
            '- \'calculate_adjective_similarities\' (bool): Calculate gender similarity scores.\n'
            '- \'select_top_adjectives\' (bool): Select top N gendered adjectives.\n'
            '- \'remove_adjective_duplicates\' (bool): Remove duplicate adjectives.\n'
            '- \'adjective_definition_review\' (bool): Generate adjective review CSVs.\n'
            '- \'remove_unwanted_adjectives\' (bool): Remove unwanted adjectives.\n'
            '- \'create_stimulus_files\' (bool): Create adjective stimulus files.\n'
            '- \'conduct_control_tests\' (bool): Conduct control tests.\n'
            '- \'conduct_experimental_tests\' (bool): Conduct experimental tests.\n')
    
    parser.add_argument('--nouns_file', type=str, default='../materials/nouns.csv', help='Path to the noun dataset CSV file.')
    
    parser.add_argument('--adjective_gender_association_method', type=str, default='cosine_similarity', help='Method for calculating gender associations.\n'
                        '- \'cosine_similarity\': Calculate raw cosine similarity between adjective and gender vectors.\n'
                        '- \'semantic_differential\': Calculate difference between adjective-gender similarity scores.')
    
    parser.add_argument('--top_n_adjectives', type=int, default=100, help='Number of top gendered adjectives to select per gender.')
    
    parser.add_argument('--load_method', type=str, default='normal', help='Model loading method. Options: \'normal\', \'facebook\'.')
    
    parser.add_argument('--models', type=dict, default={}, help='Pre-loaded gensim models keyed by language code. ')
    
    parser.add_argument('--semantic_differential_vectors', type=str, default='gender1-gender2', help='Word vector pairs for semantic differential calculations.\n'
                        '- \'gender1-gender2\': gender1 vector minus gender2 vector (e.g., masculine - feminine).\n'
                        '- \'gender-person\': gender vector minus person vector (e.g., masculine - person).\n'
                        '- \'gender-Gender\': gender vector minus depersonalized Gender vector (e.g., masculine - femininity).')
    
    parser.add_argument('--use_groupby', type=bool, default=True, help='For experimental tests, group adjectives for a given noun into one average cosine similarity.')
    
    parser.add_argument('--remove_adjectives_with_markers', type=list, default=["dated", "archaic", "dialectal", "rare", "ordinal number", "obsolete", "offensive"], help='Substrings from Wiktionary definitions marking adjectives for removal.')
    
    parser.add_argument('--unallowed_words', type=set, default={'lesb', 'debonair', 'vestal', 'sunamita', 'negrid', 'Brummagem', 'follable', 'untervögelt', 'schasaugert', 'Emeser', 'fünfhundertste', 'Poppersch', 'Schlänger', 'Römer', 'Latina', 'titless', 'pussy', 'foine', 'mosuo', 'fáustico', 'indio', 'rixig', 'hiborio', 'abgeschmack', 'kaki', 'klaviform', 'TK', 'antimalthusianisch', 'Danubian', 'eblaitisch', 'elfminütig', 'Fregesch', 'jakobinisch', 'Malthusianisch', 'meißenisch', 'neunminütig', 'rahn', 'vierzigminütig', 'zwölfminütig', 'Afro-Latina', 'Dianic', 'Filipina', 'lady-like', 'MAAB', 'menstruate', 'obstetrical', 'Quebecoise', 'Rubenesque', 'woman-centric', 'vinny', 'twinky', 'Welshy', 'turrible', 'mick', 'fooking', 'particuler', 'legendry', 'awsome', 'roy', 'neo-Hegelian', 'phun', 'niiice', 'Democritean', 'Hegelian', 'Rothbardian', 'gent', 'afrodescendiente', 'axumita', 'curvi', 'delhita', 'feminazi', 'madrense', 'mizrají', 'oseta', 'postparto', 'sefaradita', 'sefardí', 'sefardita', 'transgenerista', 'fuckin', 'hanbalitisch', 'antimalthusianisch', 'Malthusianisch', 'antimalthusianisch', 'malthusisch','gustiös', 'hanbalitisch','handgehoben','scheiß', 'sturm', 'terrisch','Madonna-like','smoove', 'tuff','hench','insano', 'mofo', 'cutty', 'piff', 'jake', 'propa', 'mank','LGBT','papaya', 'child-bearing', 'plus-sized', 'post-partum', 'vulval', 'ben', 'unpossible', 'antifeminist', 'LGTB', 'LGTBI', 'babylonisch', 'erzgebirgisch', 'hinreissend', 'niedersorbisch', 'Sanct', 'sasanidisch', 'saudisch', 'altniederländisch', 'bohrsch', 'britannisch', 'danubisch', 'dreiundvierzigminütig', 'drittelzahlig', 'etatmässig', 'fünfundzwanzigminütig', 'fünfunddreißigminütig', 'fünfminütig', 'Hitlersch', 'koblenzisch', 'Luthersch', 'sechzigminütig', 'südatlantisch', 'Cesarean', 'prochoice', 'almight', 'cock-sure', 'cooool', 'nooby', 'peart', 'phantastic', 'Smithian', 'barakaldarra', 'bartorosellista', 'cefeida', 'chilota', 'dailamita', 'estambulita', 'kábila', 'mazahua', 'ondarrutarra', 'ranjana', 'helle', 'zirkummediterran', 'südatlantisch', 'preggers', 'vajazzled', 'shite', 'steezy', 'tinhorn', 'widdly', 'afrotropical', 'apollardado', 'mijita', 'ladilla', 'gray-haired', 'heavy-set', 'middleaged', 'Jew,' 'moustached', 'African-American', 'childbearing', 'Filipina', 'Madonna-like', 'newly-wed', 'Shunamite', 'Syrophoenician', 'teen-age', 'teen-aged', 'transgendered', 'grown-ass', 'mustached', 'Caucasian', 'biracial', 'mixed-race', 'thirties', 'forties', 'clean-shaved', 'moustachioed', 'dark-skinned', 'teenaged', 'mustachioed', 'ape', 'Afroestadounidense', 'Birracial', 'Indígena', 'sexi', 'Extraconyugal', 'Israelita', 'untrew', 'cristiano', 'jóven', 'afroestadounidense', 'birracial', 'Emesener', 'baktrisch', 'israelita', '♥-lich', 'vierzigmonatig', 'währschaft', 'wolgadeutsch', 'amisch', 'dreiundfünfzigjährig', 'kraftwerkisch', 'malisch', 'Palmyrer', 'Portaner' ,'achtundvierzigmonatig', 'padre', 'hypoäolisch', 'schwatt', 'sechsunddreißigmonatig', 'israelita', 'dreißigmonatig', 'einunddreißigeckig', 'schwul', 'mannmännlich'}, help='Words to exclude from final adjective lists.')
    
    parser.add_argument('--allowed_words', type=set, default={'rascal', 'transexual'}, help='Words to retain regardless of removal markers.')

    args = parser.parse_args()
    
    run(
        args.procedures,
        args.nouns_file,
        args.adjective_gender_association_method,
        args.top_n_adjectives,
        args.load_method,
        args.models,
        args.semantic_differential_vectors,
        args.use_groupby,
        args.remove_adjectives_with_markers,
        args.unallowed_words,
        args.allowed_words
    )

if __name__ == '__main__':
    main()
