import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class dosample:
    def __init__(self, df, agemax, seed):
        self.df = df.copy()  
        self.agemax = agemax
        self.seed = seed
        self.labels = None 

    def create_sample(self):
        """Создает стратифицированную выборку по возрастным группам"""
        bins = [18, 35, 55, 65, self.agemax]
        self.labels = [f'18-34', f'35-54', f'55-64', f'65-{self.agemax}']  
        self.df['age_group'] = pd.cut(
            self.df['AGE'], 
            bins=bins, 
            labels=self.labels, 
            right=False
        )
        
        total_original = len(self.df)
        sample_size = 311
        
        strata_sizes = {}
        for group in self.labels:
            group_count = len(self.df[self.df['age_group'] == group])
            strata_size = max(1, round((group_count / total_original) * sample_size))
            strata_sizes[group] = strata_size
        
        total_sampled = sum(strata_sizes.values())
        if total_sampled != sample_size:
            largest_strata = max(strata_sizes, key=strata_sizes.get)
            strata_sizes[largest_strata] += sample_size - total_sampled
        
        sampled_data = pd.DataFrame()
        for group, size in strata_sizes.items():
            stratum = self.df[self.df['age_group'] == group]
            if len(stratum) >= size:
                sampled_stratum = stratum.sample(n=size, random_state=self.seed)
            else:
                sampled_stratum = stratum
            sampled_data = pd.concat([sampled_data, sampled_stratum])
        
        return sampled_data

    def viz_for_sample(self, sampled_data):
        total_original = len(self.df)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].hist(self.df['AGE'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Исходная выборка: Распределение возраста')
        axes[0, 0].set_xlabel('Возраст')
        axes[0, 0].set_ylabel('Частота')
        axes[0, 1].hist(sampled_data['AGE'], bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('Новая выборка: Распределение возраста')
        axes[0, 1].set_xlabel('Возраст')
        
        gender_counts_orig = self.df['SEX'].value_counts(normalize=True)
        gender_counts_samp = sampled_data['SEX'].value_counts(normalize=True)
        
        x = np.arange(len(gender_counts_orig))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, gender_counts_orig, width, label='Исходная', alpha=0.7, color='blue')
        axes[1, 0].bar(x + width/2, gender_counts_samp, width, label='Новая', alpha=0.7, color='red')
        axes[1, 0].set_title('Сравнение распределения по полу')
        axes[1, 0].set_xlabel('Пол')
        axes[1, 0].set_ylabel('Доля')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(gender_counts_orig.index)
        axes[1, 0].legend()
        
        age_group_counts_orig = self.df['age_group'].value_counts(normalize=True).sort_index()
        age_group_counts_samp = sampled_data['age_group'].value_counts(normalize=True).sort_index()
        
        x = np.arange(len(age_group_counts_orig))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, age_group_counts_orig, width, label='Исходная', alpha=0.7, color='blue')
        axes[1, 1].bar(x + width/2, age_group_counts_samp, width, label='Новая', alpha=0.7, color='red')
        axes[1, 1].set_title('Сравнение возрастных групп')
        axes[1, 1].set_xlabel('Возрастная группа')
        axes[1, 1].set_ylabel('Доля')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(age_group_counts_orig.index)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        print("\nПроверка репрезентативности:")
        print(f"Оригинальная выборка (n={total_original}):")
        print(f"- Доля мужчин: {gender_counts_orig.get(1, 0):.1%}")
        print(f"- Возрастные группы:")
        for group in self.labels:
            prop = age_group_counts_orig.get(group, 0)
            print(f"  {group}: {prop:.1%}")
        
        print(f"\nНовая выборка (n={len(sampled_data)}):")
        print(f"- Доля мужчин: {gender_counts_samp.get(1, 0):.1%}")
        print(f"- Возрастные группы:")
        for group in self.labels:
            prop = age_group_counts_samp.get(group, 0)
            print(f"  {group}: {prop:.1%}")

    def viz_extra_features(self,sampled_data):
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        additional_vars = {'EDU':'образование',
                   'DOHOD': 'доход', 
                   'PROF': 'профессия', 
                   'FO':'федеральный округ', 
                   'TIP':'тип населенного пункта'}
        n_vars = len(additional_vars)
        axes_flat = axes.flatten()

        for i, var in enumerate(additional_vars):
            orig_prop = self.df[var].value_counts(normalize=True).sort_index()
            sample_prop = sampled_data[var].value_counts(normalize=True).sort_index()
            all_categories = sorted(set(orig_prop.index) | set(sample_prop.index))
            compare_df = pd.DataFrame({
                'Original': orig_prop.reindex(all_categories).fillna(0),
                'Sample': sample_prop.reindex(all_categories).fillna(0)
            })
    
            compare_df.plot.bar(ax=axes_flat[i], rot=45, alpha=0.8)
            axes_flat[i].set_title(f'Распределение: {var} ({additional_vars[var]})')
            axes_flat[i].set_ylabel('Доля')
            axes_flat[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        for j in range(n_vars, len(axes_flat)):
            axes_flat[j].axis('off')
            plt.tight_layout()
            plt.show()