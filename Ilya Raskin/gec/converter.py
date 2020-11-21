from lxml import html
import os


class CoNLLConverter():
    def get_target(self, s_text, annotations):
        t_text = ''
        p = 0
        o = 0

        for (sp, so, ep, eo, correction) in annotations:
            if sp > p:
                t_text += s_text[p][o:] + '\n'
                for i in range(p + 1, sp):
                    t_text += s_text[i] + '\n'
                p, o = sp, 0
            t_text += s_text[p][o:so] + correction
            p, o = ep, eo
            if t_text[-1] == ' ' and o < len(s_text[p]) and s_text[p][o] == ' ': # Вроде, всегда верно, но мало ли...
                o += 1 # Чтобы после удаления слова не образовывалось 2 пробела подряд

        t_text += s_text[p][o:] + '\n'
        for i in range(p + 1, len(s_text)):
            t_text += s_text[i] + '\n'

        return t_text          
    
    def prepare_one_doc(self, doc):
        root = html.fromstring(doc)
        text = root.xpath('//doc/text')[0]
        s_text = [i for i in text.text_content().split('\n') if i]
        annotations = []
        prev_ep, prev_eo = -1, -1

        for m in root.xpath('//doc/annotation/mistake'):
            sp, so, ep, eo = int(m.get('start_par')), int(m.get('start_off')), int(m.get('end_par')), int(m.get('end_off'))
            correction = m.xpath('correction')[0].text
            correction = correction if correction else ''
            if prev_ep == sp and prev_eo == so: # Ошибка датасета. Конец ошибки и начало следующей не могут совпадать
                 sp_, so_, ep_, eo_, correction_ = annotations[-1]
                 annotations[-1] = (sp_, so_, ep_, eo_ - 1, correction_)
            prev_ep, prev_eo = ep, eo
            annotations.append((sp, so, ep, eo, correction))

        return '\n'.join(s_text) + '\n', self.get_target(s_text, annotations)

    def prepare(self, data_file, source_file, target_file):
        source = open(source_file, 'w')
        target = open(target_file, 'w')
        with open(data_file, 'r') as f:
            doc = ''
            for line in f:
                doc += line
                if '</DOC>' in line:
                    s_text, t_text = self.prepare_one_doc(doc)
                    source.write(s_text)
                    target.write(t_text)
                    doc = ''
        source.close()
        target.close()
        
