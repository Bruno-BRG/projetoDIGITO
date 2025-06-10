#!/usr/bin/env python3
"""
Teste Específico: Salvamento do Modelo após Fine-tuning
Verifica se o modelo é persistido no disco após aprendizado por reforço
"""

import os
import numpy as np
import time
from datetime import datetime
from main import FeedbackManager
from trainer import MNISTTrainer

def test_model_save_after_finetune():
    """Testa se o modelo é salvo após fine-tuning com feedback"""
    print("🔍 Testando Salvamento do Modelo após Fine-tuning...")
    
    # 1. Backup do modelo atual (se existir)
    model_path = "mnist_model.pth"
    backup_path = "mnist_model_backup.pth"
    
    if os.path.exists(model_path):
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"📁 Backup criado: {backup_path}")
    
    # 2. Obter timestamp do arquivo antes do fine-tuning
    if os.path.exists(model_path):
        timestamp_before = os.path.getmtime(model_path)
        size_before = os.path.getsize(model_path)
        print(f"📊 Modelo antes: {size_before} bytes, modificado em {datetime.fromtimestamp(timestamp_before)}")
    else:
        timestamp_before = 0
        size_before = 0
        print("⚠️ Nenhum modelo encontrado inicialmente")
    
    # 3. Criar dados de feedback sintéticos
    feedback_manager = FeedbackManager("test_model_save.json")
    trainer = MNISTTrainer()
    
    # Carregar modelo se existir
    if os.path.exists(model_path):
        trainer.load_model(model_path)
        print("✅ Modelo carregado para teste")
    
    # Criar múltiplos feedbacks sintéticos
    fake_feedbacks = []
    for i in range(10):  # 10 exemplos para garantir fine-tuning
        fake_image = np.random.rand(28, 28).astype(np.float32)
        predicted_digit = i % 10
        actual_digit = (i + 1) % 10  # Força erro para simular correção
        
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "image": fake_image.tolist(),
            "predicted_label": predicted_digit,
            "actual_label": actual_digit,
            "confidence": 0.7,
            "correct": predicted_digit == actual_digit
        }
        fake_feedbacks.append(feedback_entry)
    
    print(f"📝 Criados {len(fake_feedbacks)} feedbacks sintéticos")
    
    # 4. Executar fine-tuning
    print("🎯 Iniciando fine-tuning...")
    success = trainer.fine_tune_with_feedback(fake_feedbacks, epochs=2, lr=0.0001)
    
    # 5. Verificar se o modelo foi salvo
    time.sleep(1)  # Aguardar um pouco para garantir que o arquivo foi escrito
    
    if os.path.exists(model_path):
        timestamp_after = os.path.getmtime(model_path)
        size_after = os.path.getsize(model_path)
        print(f"📊 Modelo depois: {size_after} bytes, modificado em {datetime.fromtimestamp(timestamp_after)}")
        
        # Verificar se o arquivo foi modificado
        if timestamp_after > timestamp_before:
            print("✅ SUCESSO: Modelo foi salvo após fine-tuning!")
            print(f"🕐 Diferença de tempo: {timestamp_after - timestamp_before:.2f} segundos")
            
            # Verificar se o tamanho mudou (pode ou não mudar)
            if size_after != size_before:
                print(f"📏 Tamanho alterado: {size_before} → {size_after} bytes")
            else:
                print("📏 Tamanho mantido (normal para fine-tuning)")
            
            return True
        else:
            print("❌ FALHA: Modelo NÃO foi salvo após fine-tuning!")
            print(f"🕐 Timestamps: antes={timestamp_before}, depois={timestamp_after}")
            return False
    else:
        print("❌ FALHA: Arquivo do modelo não existe após fine-tuning!")
        return False
    
    # 6. Limpeza
    if os.path.exists("test_model_save.json"):
        os.remove("test_model_save.json")
    
    # Restaurar backup se algo deu errado
    if not success and os.path.exists(backup_path):
        import shutil
        shutil.copy2(backup_path, model_path)
        print("🔄 Modelo original restaurado do backup")
    
    # Remover backup
    if os.path.exists(backup_path):
        os.remove(backup_path)

def test_model_persistence():
    """Testa se o modelo salvo pode ser carregado novamente"""
    print("\n🔍 Testando Persistência do Modelo...")
    
    model_path = "mnist_model.pth"
    
    if not os.path.exists(model_path):
        print("❌ Modelo não encontrado para teste de persistência")
        return False
    
    # Carregar modelo
    trainer1 = MNISTTrainer()
    success1 = trainer1.load_model(model_path)
    
    if not success1:
        print("❌ Falha ao carregar modelo")
        return False
    
    # Fazer uma predição
    test_image = np.random.rand(28, 28).astype(np.float32)
    pred1, conf1 = trainer1.predict(test_image)
    
    # Carregar em uma nova instância
    trainer2 = MNISTTrainer()
    success2 = trainer2.load_model(model_path)
    
    if not success2:
        print("❌ Falha ao carregar modelo na segunda instância")
        return False
    
    # Fazer a mesma predição
    pred2, conf2 = trainer2.predict(test_image)
    
    # Verificar se as predições são consistentes
    if pred1 == pred2 and abs(conf1 - conf2) < 0.001:
        print("✅ SUCESSO: Modelo persistido corretamente!")
        print(f"📊 Predição consistente: {pred1} (confiança: {conf1:.3f})")
        return True
    else:
        print("❌ FALHA: Predições inconsistentes entre carregamentos!")
        print(f"📊 Pred1: {pred1} ({conf1:.3f}) vs Pred2: {pred2} ({conf2:.3f})")
        return False

def main():
    """Executa todos os testes de salvamento"""
    print("🧪 === TESTE DE SALVAMENTO DO MODELO APÓS RL ===\n")
    
    tests = [
        ("Salvamento após Fine-tuning", test_model_save_after_finetune),
        ("Persistência do Modelo", test_model_persistence)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro no teste {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n🏁 === RESULTADOS DOS TESTES ===")
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 TODOS OS TESTES DE SALVAMENTO PASSARAM!")
        print("💾 O modelo está sendo salvo corretamente após fine-tuning!")
    else:
        print("\n⚠️ ALGUNS TESTES FALHARAM")
        print("🔧 Verificar problemas de salvamento do modelo")

if __name__ == "__main__":
    main()
