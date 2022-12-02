from transformers import TFBertModel, TFRobertaModel
import tensorflow as tf
import numpy as np

MAX_LEN = 250
#cls_token_activate=False, num_epochs=1, initial_learning_rate=None, end_learning_rate=None,
def build_model(config, embeddings='bert-base-uncased', schedule=None, join='cosine'):

  if embeddings == 'bert-base-uncased' or embeddings == 'bert-large-uncased':
    encode_input = TFBertModel.from_pretrained(embeddings, output_attentions=False, use_cache=False) #.bert
  if embeddings == 'roberta-base' or embeddings == 'roberta-large-uncased':
    encode_input = TFRobertaModel.from_pretrained(embeddings, output_attentions=False) #.bert

  toks1=tf.keras.Input(shape=(MAX_LEN,), dtype='int32')
  atts1=tf.keras.Input(shape=(MAX_LEN,), dtype="int32")
  out1=encode_input(input_ids=toks1,attention_mask=atts1)

  toks2=tf.keras.Input(shape=(MAX_LEN,), dtype='int32')
  atts2=tf.keras.Input(shape=(MAX_LEN,), dtype="int32")
  out2=encode_input(input_ids=toks2,attention_mask=atts2)

  try:
    if (embeddings == 'roberta-base' or embeddings == 'roberta-large-uncased') and config['cls_token_activate']:
      return "ERROR: can't use CLS token with RoBERTa model"

    if (config['cls_token_activate']):  
      emb1 = out1[0][:,0,:]
      emb2 = out2[0][:,0,:]
    else:
      emb1=tf.reduce_mean(out1[0],1)
      emb2=tf.reduce_mean(out2[0],1)
  except:
    print("WARNING: config['cls_token_activate'] not defined")
    emb1=tf.reduce_mean(out1[0],1)
    emb2=tf.reduce_mean(out2[0],1)
    print("Mean strategy set by default")
  

  if join == 'cosine':
    preds = build_cosine(emb1, emb2)
  elif join == 'softmax':
    preds = build_softmax(emb1, emb2)
  else:
    return 'ERROR'


  if schedule is None:
    if config['lr'][0] is None:
      return 'ERROR: no learning rate or schedule'
    
    else:
      if config['lr'][1] is None:
        config['lr'][1] = config['lr'][0]
      schedule = build_schedule(config['lr'][0],config['lr'][1],config['num_epochs'])

  model = tf.keras.Model(inputs=[toks1,atts1,toks2,atts2], outputs=preds)

  model.compile(
    loss = 'mse',
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule),
    metrics = ['mse'])

  print(model.summary())

  return model


def lr_warmup_cosine_decay(global_step,
                          warmup_steps,
                          hold = 0,
                          total_steps=0,
                          start_lr=0.0,
                          target_lr=1e-3):
  # Cosine decay
  # There is no tf.pi so we wrap np.pi as a TF constant
  learning_rate = 0.5 * target_lr * (1 + tf.cos(tf.constant(np.pi) * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

  # Target LR * progress of warmup (=1 at the final warmup step)
  warmup_lr = target_lr * (global_step / warmup_steps)

  # Choose between warmup_lr, target_lr and learning_rate based on whether global_step < warmup_steps and we're still holding.
  # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
  if hold > 0:
      learning_rate = tf.where(global_step > warmup_steps + hold,
                                learning_rate, target_lr)
  
  learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
  return learning_rate

class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def init(self, start_lr, target_lr, warmup_steps, total_steps, hold):
        super().init()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def call(self, step):
        lr = lr_warmup_cosine_decay(global_step=step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)

        return tf.where(
            step > self.total_steps, 0.0, lr, name="learning_rate"
        )

def build_schedule(initial_learning_rate,end_learning_rate,num_epochs, number_of_samples=1290, warmup=False, warmup_value=0.1, power_decay=1):
  num_train_steps = number_of_samples * num_epochs
  
  if warmup:
    warmup_steps = int(warmup_value*num_train_steps) #0.05
    schedule = WarmUpCosineDecay(start_lr=0, target_lr=initial_learning_rate, warmup_steps=warmup_steps, total_steps=num_train_steps, hold=warmup_steps)
  
  else:
    schedule = tf.keras.optimizers.schedules.PolynomialDecay(
          initial_learning_rate=initial_learning_rate,
          end_learning_rate=end_learning_rate,
          power=power_decay,
          decay_steps=num_train_steps)
  
  return schedule


def build_cosine(emb1, emb2):
  cosine_similarity = tf.keras.layers.Dot(axes=1,normalize=True, dtype="float16")
  preds = cosine_similarity([emb1,emb2])
  
  return preds

def build_softmax(emb1, emb2):

  sub_emb =  tf.keras.layers.subtract([emb1, emb2])
  concat = tf.keras.layers.Concatenate(axis=-1)([emb1, emb2, sub_emb])

  preds = tf.keras.layers.Dense(2, activation="softmax")(concat)
  
  return preds

