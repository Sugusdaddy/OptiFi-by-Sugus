import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DeFiPortfolioOptimizer:
    def __init__(self, risk_free_rate=0.05):
        self.risk_free_rate = risk_free_rate
        self.prices = None
        self.returns = None
        self.ef = None
        
    def fetch_data(self, tickers, period='1y'):
        """
        Descarga datos históricos de Yahoo Finance para cada activo individualmente.
        """
        print(f"📊 Descargando datos para {len(tickers)} activos ({period})...")
        
        prices_dict = {}
        
        for ticker in tickers:
            try:
                print(f"  ⏳ Descargando {ticker}...", end=" ")
                data = yf.download(ticker, period=period, progress=False, interval='1d')
                
                if data.empty:
                    print(f"⚠️  Sin datos")
                    continue
                
                # Usar 'Close' o 'Adj Close' según disponibilidad
                if 'Adj Close' in data.columns:
                    prices_dict[ticker] = data['Adj Close']
                elif 'Close' in data.columns:
                    prices_dict[ticker] = data['Close']
                else:
                    print(f"❌ No hay columna de precio")
                    continue
                    
                print(f"✅ ({len(data)} registros)")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        if not prices_dict:
            raise ValueError("❌ No se pudieron descargar datos para ningún activo")
        
        # Crear DataFrame con todos los precios, alineando índices
        self.prices = pd.concat(prices_dict, axis=1)
        self.prices.columns = list(prices_dict.keys())
        self.prices = self.prices.dropna()
        
        print(f"\n✅ Datos cargados exitosamente")
        print(f"   Período: {self.prices.index[0].date()} a {self.prices.index[-1].date()}")
        print(f"   Activos: {list(self.prices.columns)}")
        print(f"   Registros: {len(self.prices)} días\n")
        
        # Calcular retornos diarios
        self.returns = self.prices.pct_change().dropna()
        
        return self.prices
    
    def calculate_portfolio_stats(self, weights):
        """Calcula retorno, volatilidad y Sharpe ratio para pesos dados."""
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights, self.returns.cov() * 252 @ weights))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
        return portfolio_return, portfolio_std, sharpe
    
    def optimize_portfolio(self, constraint_min=0.05):
        """
        Optimiza la cartera usando EfficientFrontier (maximiza Sharpe ratio).
        constraint_min: peso mínimo por activo
        """
        print("⚙️  Optimizando cartera...")
        
        # Calcular matriz de covarianza y retornos esperados
        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        # Crear frontera eficiente
        self.ef = EfficientFrontier(
            mean_returns, 
            cov_matrix, 
            weight_bounds=(constraint_min, 0.4)  # Min 5%, Max 40% por activo
        )
        
        # Maximizar Sharpe ratio
        weights = self.ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        
        return weights
    
    def display_results(self, weights):
        """Muestra resultados formateados."""
        ret, vol, sharpe = self.ef.portfolio_performance(verbose=False)
        
        print("=" * 60)
        print("📈 CARTERA ÓPTIMA (MÁXIMO SHARPE RATIO)")
        print("=" * 60)
        print(f"\n💰 Pesos de Inversión:\n")
        
        for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            weight_pct = weight * 100
            bar = "█" * int(weight_pct / 2)
            print(f"  {ticker:10s} {weight_pct:6.2f}% {bar}")
        
        print(f"\n📊 Métricas de Rendimiento:")
        print(f"  Retorno Esperado:     {ret*100:7.2f}%")
        print(f"  Volatilidad (Riesgo): {vol*100:7.2f}%")
        print(f"  Ratio de Sharpe:      {sharpe:7.3f}")
        print(f"  Tasa Libre de Riesgo: {self.risk_free_rate*100:5.1f}%")
        print("=" * 60)
        
        return ret, vol, sharpe
    
    def plot_efficient_frontier(self, weights, num_portfolios=5000):
        """Visualiza la frontera eficiente y la cartera óptima."""
        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        print("\n📊 Generando gráfico de frontera eficiente...")
        
        # Generar carteras aleatorias
        results = np.zeros((3, num_portfolios))
        for i in range(num_portfolios):
            w = np.random.random(len(self.returns.columns))
            w /= np.sum(w)
            
            ret = np.sum(mean_returns * w)
            std = np.sqrt(np.dot(w, cov_matrix @ w))
            sharpe = (ret - self.risk_free_rate) / std
            
            results[0,i] = std
            results[1,i] = ret
            results[2,i] = sharpe
        
        # Cartera óptima
        opt_ret, opt_std, opt_sharpe = self.ef.portfolio_performance(verbose=False)
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(12, 7))
        scatter = ax.scatter(
            results[0,:]*100, results[1,:]*100, 
            c=results[2,:], cmap='viridis', alpha=0.5, s=20
        )
        
        # Marcar cartera óptima
        ax.scatter(opt_std*100, opt_ret*100, marker='*', color='red', 
                  s=800, edgecolors='darkred', linewidth=2, 
                  label=f'Cartera Óptima (Sharpe: {opt_sharpe:.3f})', zorder=5)
        
        ax.set_xlabel('Volatilidad (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Retorno Esperado (%)', fontsize=12, fontweight='bold')
        ax.set_title('Frontera Eficiente: Carteras Óptimas Multi-Activos', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='upper left')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Ratio de Sharpe', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def allocate_capital(self, weights, capital=10000):
        """Convierte pesos en asignación de capital discreto."""
        print(f"\n💵 Asignación de Capital (${capital:,}):\n")
        
        allocation = DiscreteAllocation(
            weights, 
            self.prices.iloc[-1], 
            total_portfolio_value=capital
        )
        
        discrete_allocation, leftover = allocation.greedy_portfolio()
        
        total_invested = 0
        for ticker, shares in sorted(discrete_allocation.items()):
            price = self.prices[ticker].iloc[-1]
            value = shares * price
            total_invested += value
            print(f"  {ticker:10s} {shares:5.0f} unidades @ ${price:10.2f} = ${value:10.2f}")
        
        print(f"\n  {'Efectivo restante:':30s} ${leftover:10.2f}")
        print(f"  {'Total invertido:':30s} ${total_invested:10.2f}")


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # 1. Definir activos: acciones + criptomonedas
    assets = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD']
    
    # 2. Crear optimizador
    optimizer = DeFiPortfolioOptimizer(risk_free_rate=0.05)
    
    try:
        # 3. Descargar datos (últimos 2 años)
        optimizer.fetch_data(assets, period='2y')
        
        # 4. Optimizar cartera
        optimal_weights = optimizer.optimize_portfolio()
        
        # 5. Mostrar resultados
        optimizer.display_results(optimal_weights)
        
        # 6. Visualizar frontera eficiente
        optimizer.plot_efficient_frontier(optimal_weights)
        
        # 7. Asignación de capital (ejemplo: $10,000)
        optimizer.allocate_capital(optimal_weights, capital=10000)
        
    except Exception as e:
        print(f"❌ Error en la ejecución: {e}")
        import traceback
        traceback.print_exc()