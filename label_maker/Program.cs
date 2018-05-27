using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace label_maker
{
    class Program
    {
        static void Main(string[] args)
        {
            List<string> train_list = new List<string>();
            List<string> val_list = new List<string>();
            Random rd = new Random(73451);
            DirectoryInfo father = new DirectoryInfo(@"D:\Research\CNN-based-Indoor-Localization\data\train");
            DirectoryInfo[] children = father.GetDirectories("*.*", SearchOption.TopDirectoryOnly);
            foreach (var child in children)
            {
                FileInfo[] files = child.GetFiles("*.jpg", SearchOption.TopDirectoryOnly);
                foreach (var file in files)
                {
                    if (rd.Next(0,100)>=5)
                    {
                        // train
                        train_list.Add(file.FullName + " " + child.Name);
                    }
                    else
                    {
                        // val
                        val_list.Add(file.FullName + " " + child.Name);
                    }
                }
            }
            Shuffle(train_list);
            Shuffle(val_list);
            SaveList(train_list, @"D:\Research\CNN-based-Indoor-Localization\label\train_label.txt");
            SaveList(val_list, @"D:\Research\CNN-based-Indoor-Localization\label\val_label.txt");
        }

        public static void Shuffle<T>(IList<T> list)
        {
            Random rng = new Random();
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static void SaveList(List<string> lists, string savepath)
        {
            FileStream fs = new FileStream(savepath, FileMode.Append);
            StreamWriter sw = new StreamWriter(fs);
            for (int i = 0; i < lists.Count; i++)
            {
                sw.WriteLine(lists[i]);
                sw.Flush();
            }
            sw.Close();
            fs.Close();
        }
    }
}
