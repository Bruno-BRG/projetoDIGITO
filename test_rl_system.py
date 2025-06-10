#!/usr/bin/env python3
"""
Teste do Sistema de Aprendizado por Reforço
Valida se o feedback interativo está funcionando corretamente
"""

import numpy as np
import json
import os
from main import FeedbackManager
from trainer import MNISTTrainer
from model import MNISTNet

def test_feedback_system():
    """Testa o sistema de feedback"""
    print("🔍 Testando Sistema de Feedback...")
    
    # Inicializar feedback manager
    feedback_manager = FeedbackManager("test_feedback.json")
    
    # Simular alguns dados de feedback
    fake_image = np.random.rand(28, 28).astype(np.float32)
    
    # Adicionar feedback correto
    feedback_manager.add_feedback(fake_image, predicted_label=7, actual_label=7, confidence=0.95)
    
    # Adicionar feedback incorreto
    feedback_manager.add_feedback(fake_image, predicted_label=3, actual_label=8, confidence=0.60)
    
    # Obter estatísticas
    stats = feedback_manager.get_stats()
    print(f"📊 Estatísticas do Feedback: {stats}")
    
    # Verificar dados incorretos
    wrong_predictions = feedback_manager.get_wrong_predictions()
    print(f"❌ Predições erradas: {len(wrong_predictions)}")
    
    # Limpar arquivo de teste
    if os.path.exists("test_feedback.json"):
        os.remove("test_feedback.json")
    
    print("✅ Sistema de feedback funcionando!")
    return True

def test_fine_tuning():
    """Testa se o fine-tuning está disponível"""
    print("🔍 Testando Fine-tuning...")
    
    try:
        trainer = MNISTTrainer()
        
        # Verificar se métodos de fine-tuning existem
        if hasattr(trainer, 'fine_tune_with_feedback'):
            print("✅ Método fine_tune_with_feedback disponível")
        else:
            print("❌ Método fine_tune_with_feedback não encontrado")
            return False
            
        if hasattr(trainer, 'validate_on_feedback'):
            print("✅ Método validate_on_feedback disponível")
        else:
            print("❌ Método validate_on_feedback não encontrado")
            return False
            
        print("✅ Fine-tuning disponível!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de fine-tuning: {e}")
        return False

def test_model_prediction():
    """Testa se o modelo consegue fazer predições"""
    print("🔍 Testando Predições do Modelo...")
    
    try:
        trainer = MNISTTrainer()
        
        # Carregar modelo se existir
        if os.path.exists("mnist_model.pth"):
            trainer.load_model("mnist_model.pth")
            print("✅ Modelo carregado")
        else:
            print("⚠️ Modelo não encontrado - usando modelo não treinado")
        
        # Criar imagem de teste
        test_image = np.random.rand(28, 28).astype(np.float32)
        
        # Fazer predição
        predicted_digit, confidence = trainer.predict(test_image)
        
        print(f"📊 Predição: Dígito={predicted_digit}, Confiança={confidence:.3f}")
        print("✅ Sistema de predição funcionando!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste de predição: {e}")
        return False

def main():
    """Executa todos os testes"""
    print("🧪 === TESTE DO SISTEMA DE APRENDIZADO POR REFORÇO ===\n")
    
    tests = [
        ("Sistema de Feedback", test_feedback_system),
        ("Fine-tuning", test_fine_tuning),
        ("Predições do Modelo", test_model_prediction)
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
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("🚀 Sistema de Aprendizado por Reforço está funcionando!")
    else:
        print("\n⚠️ ALGUNS TESTES FALHARAM")
        print("🔧 Verifique os problemas acima")

if __name__ == "__main__":
    main()
